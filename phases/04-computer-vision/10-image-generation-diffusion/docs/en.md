# Generowanie obrazów — Modele dyfuzyjne

> Model dyfuzyjny uczy się usuwać szum. Naucz go usuwać niewielką ilość szumu z obrazu z szumem, powtórz to wstecz tysiąc razy, a otrzymasz generator obrazów.

**Typ:** Zbuduj
**Języki:** Python
**Wymagania wstępne:** Faza 4 Lekcja 07 (U-Net), Faza 1 Lekcja 06 (Prawdopodobieństwo), Faza 3 Lekcja 06 (Optymizatory)
**Szacowany czas:** ~75 minut

## Cele uczenia się

- Wyprowadź proces szumu forward `x_0 -> x_1 -> ... -> x_T` i wyjaśnij, dlaczego zamknięta forma `q(x_t | x_0)` obowiązuje dla dowolnego t
- Zaimplementuj cel treningowy w stylu DDPM, który regresuje szum dodany na każdym kroku, oraz sampler, który wraca od czystego szumu do obrazu
- Zbuduj U-Net z warunkowaniem czasowym (na tyle mały, by trenować na CPU), który przewiduje szum dla dowolnego kroku czasowego
- Wyjaśnij różnicę między próbkowaniem DDPM i DDIM oraz kiedy każde z nich jest odpowiednie (Lekcja 23 dogłębnie omawia flow matching i rectified flow)

## Problem

GANy generują jednorazowo: szum w, obraz out, jeden przebieg forward. Są szybkie i trudne do trenowania. Modele dyfuzyjne generują iteracyjnie: start od czystego szumu, usuwanie szumu małymi krokami, obraz się wyłania. Są wolne i łatwe do trenowania. Przez ostatnie pięć lat dominowała ta druga właściwość: każdy mały zespół może trenować model dyfuzyjny i otrzymać rozsądne próbki; trening GAN to rzemiosło, którego uczysz się przez lata nieudanych uruchomień.

Poza stabilnością treningu, iteracyjna struktura dyfuzji jest tym, co odblokowuje wszystko, co robi nowoczesna generacja obrazów: warunkowanie tekstowe, inpainting, edycja obrazu, super-rozdzielczość, kontrolowany styl. Każdy krok pętli próbkowania to miejsce do wstrzyknięcia nowego ograniczenia. To jest powód, dla którego Stable Diffusion, Imagen, DALL-E 3, Midjourney i każdy model obrazu z kontrolą, którego będziesz używać, są wszystkie oparte na dyfuzji.

Ta lekcja buduje minimalny DDPM: szum forward, usuwanie szumu backward, pętla treningowa. Następna lekcja (Stable Diffusion) podłączy go do systemu produkcyjnego z VAE, koderem tekstowym i classifier-free guidance.

## Koncepcja

### Proces forward

Weź obraz `x_0`. Dodaj odrobinę szumu Gaussowskiego, aby otrzymać `x_1`. Dodaj jeszcze trochę, aby otrzymać `x_2`. Kontynuuj przez T kroków, aż `x_T` będzie prawie nieodróżnialne od czystego szumu Gaussowskiego.

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1},  beta_t * I)
```

`beta_t` to mały harmonogram wariancji, typowo liniowy od 0.0001 do 0.02 przez T=1000 kroków. Każdy krok nieznacznie zmniejsza sygnał i wstrzykuje świeży szum.

### Zamknięta forma skoku

Dodawanie szumu krok po kroku jest łańcuchem Markowa, ale matematyka się składa: możesz spróbkować `x_t` bezpośrednio z `x_0` w jednym kroku.

```
Define alpha_t = 1 - beta_t
Define alpha_bar_t = prod_{s=1..t} alpha_s

Then:
  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I)

Equivalently:
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  where epsilon ~ N(0, I)
```

To pojedyncze równanie jest całym powodem, dla którego dyfuzja jest praktyczna. Podczas treningu wybierasz losowe `t`, spróbkuj `x_t` bezpośrednio z `x_0` i trenuj w jednym kroku — bez symulacji pełnego łańcucha Markowa.

### Proces odwrotny

Proces forward jest ustalony. Proces odwrotny `p(x_{t-1} | x_t)` jest tym, czego uczy się sieć neuronowa. Modele dyfuzyjne nie przewidują `x_{t-1}` bezpośrednio; przewidują szum `epsilon` dodany w kroku t, a matematyka wyprowadza `x_{t-1}` z niego.

```mermaid
flowchart LR
    X0["x_0<br/>(czysty obraz)"] --> Q1["q(x_t|x_0)<br/>dodaj szum"]
    Q1 --> XT["x_t<br/>(z szumem)"]
    XT --> MODEL["model(x_t, t)"]
    MODEL --> EPS["przewidziany epsilon"]
    EPS --> LOSS["MSE względem<br/>prawdziwego epsilon"]

    XT -.->|próbkowanie| STEP["p(x_{t-1}|x_t)"]
    STEP -.-> XT1["x_{t-1}"]
    XT1 -.->|powtórz 1000x| X0S["x_0 (spróbkowany)"]

    style X0 fill:#dcfce7,stroke:#16a34a
    style MODEL fill:#fef3c7,stroke:#d97706
    style LOSS fill:#fecaca,stroke:#dc2626
    style X0S fill:#dbeafe,stroke:#2563eb
```

### Funkcja straty treningowej

Dla każdego kroku treningowego:

1. Spróbkuj prawdziwy obraz `x_0`.
2. Spróbkuj krok czasowy `t` równomiernie z [1, T].
3. Spróbkuj szum `epsilon ~ N(0, I)`.
4. Oblicz `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`.
5. Przewiduj `epsilon_theta(x_t, t)` za pomocą sieci.
6. Zminimalizuj `|| epsilon - epsilon_theta(x_t, t) ||^2`.

To wszystko. Sieć neuronowa uczy się przewidywać szum na dowolnym kroku czasowym. Funkcja straty to MSE. Nie ma gry adversarial, nie ma kolapsu, nie ma oscylacji.

### Sampler (DDPM)

Aby generować: startuj od `x_T ~ N(0, I)` i wracaj wstecz krok po kroku.

```
for t = T, T-1, ..., 1:
    eps = model(x_t, t)
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sqrt(beta_t) * z
    where z ~ N(0, I) if t > 1, else 0
return x_0
```

Kluczowe jest to, że chociaż wsteczny warunek nie jest znany w zamkniętej formie ogólnie, dla tego konkretnego procesu forward Gaussowskiego jest. Brzydko wyglądające współczynniki to to, co daje ci reguła Bayesa.

### Dlaczego 1000 kroków

Harmonogram szumu forward jest wybrany tak, że każdy krok dodaje wystarczająco dużo szumu, że krok wsteczny jest prawie Gaussowski. Zbyt mało kroków i krok wsteczny jest daleki od Gaussowskiego, sieć nie może go dobrze modelować. Zbyt wiele kroków i próbkowanie staje się drogie z malejącym zyskiem. T=1000 z harmonogramem liniowym to domyślne DDPM.

### DDIM: 20x szybsze próbkowanie

Trening jest taki sam. Próbkowanie się zmienia. DDIM (Song et al., 2020) definiuje deterministyczny proces wsteczny, który pomija kroki czasowe bez przeuczania. Próbkowanie w 50 krokach z DDIM daje jakość bliską DDPM z 1000 kroków. Każdy system produkcyjny używa DDIM lub jeszcze szybszego wariantu (DPM-Solver, Euler ancestral).

### Warunkowanie czasowe

Sieć `epsilon_theta(x_t, t)` musi wiedzieć, który krok czasowy usuwa szum. Nowoczesne modele dyfuzyjne wstrzykują `t` przez sinusoidalne time embeddings (ten sam pomysł co positional encoding w transformerach), które są dodawane do feature maps na każdym poziomie U-Net.

```
t_embedding = sinusoidal(t)
feature_map += MLP(t_embedding)
```

Bez warunkowania czasowego sieć musi zgadywać poziom szumu z samego obrazu, co działa, ale jest o wiele mniej efektywne pod względem próbek.

## Zbuduj to

### Krok 1: Harmonogram szumu

```python
import torch

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
    }

schedule = precompute_schedule(linear_beta_schedule(T=1000))
```

Wylicz raz, pobierz przez indeks podczas treningu i próbkowania.

### Krok 2: Dyfuzja forward (q_sample)

```python
def q_sample(x0, t, noise, schedule):
    sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise
```

Jednoliniowa zamknięta forma. `t` to batch kroków czasowych, po jednym na obraz w batchu.

### Krok 3: Mały U-Net z warunkowaniem czasowym

```python
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base=32, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )
        self.t_dim = t_dim
        self.enc1 = nn.Conv2d(img_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base * 2, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(base * 4, base * 2)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1)) + t_proj
        h3 = F.silu(self.mid(h2))
        d1 = F.silu(self.dec1(h3))
        d2 = torch.cat([d1, h1], dim=1)
        return self.dec2(d2)
```

Dwupoziomowy U-Net z warunkowaniem czasowym wstrzykniętym w bottleneck. Skaluj w górę głębokość i szerokość dla prawdziwych obrazów.

### Krok 4: Pętla treningowa

```python
def train_step(model, x0, schedule, optimizer, device, T=1000):
    model.train()
    x0 = x0.to(device)
    bs = x0.size(0)
    t = torch.randint(0, T, (bs,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

To jest cała pętla treningowa. Nie ma gry GAN, nie ma specjalistycznej funkcji straty, jedno wywołanie MSE.

### Krok 5: Sampler (DDPM)

```python
@torch.no_grad()
def sample(model, schedule, shape, T=1000, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    betas = schedule["betas"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        coef = betas[t] / sqrt_one_minus_a[t]
        mean = sqrt_recip_alphas[t] * (x - coef * eps)
        if t > 0:
            x = mean + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mean
    return x
```

1000 przebiegów forward, aby wyprodukować jedną partię próbek. W prawdziwym kodzie zamienisz to na sampler DDIM 50 kroków.

### Krok 6: Sampler DDIM (deterministyczny, ~20x szybszy)

```python
@torch.no_grad()
def sample_ddim(model, schedule, shape, steps=50, T=1000, device="cpu", eta=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t = ts[i]
        t_prev = ts[i + 1]
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma ** 2) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x
```

`eta=0` jest w pełni deterministyczny (ten sam szum wejściowy zawsze produkuje ten sam wynik). `eta=1` odzyskuje DDPM.

## Użyj tego

Do pracy produkcyjnej użyj `diffusers`:

```python
from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2)
scheduler = DDPMScheduler(num_train_timesteps=1000)
```

Biblioteka dostarcza gotowe schedulery (DDPM, DDIM, DPM-Solver, Euler, Heun), konfigurowalne U-Nety, pipeline'y do text-to-image i image-to-image, oraz helpery do fine-tuningu LoRA.

Do badań, `k-diffusion` (Katherine Crowson) ma najwierniejsze implementacje referencyjne i najlepsze warianty próbkowania.

## Dostarcz to

Ta lekcja produkuje:

- `outputs/prompt-diffusion-sampler-picker.md` — prompt, który wybiera DDPM / DDIM / DPM-Solver / Euler na podstawie docelowej jakości, budżetu latency i typu warunkowania.
- `outputs/skill-noise-schedule-designer.md` — skill, który produkuje liniowy, cosinusowy lub sigmoidalny harmonogram beta dla T i docelowego poziomu korupcji, plus wykresy diagnostyczne stosunku sygnału do szumu w czasie.

## Ćwiczenia

1. **(Łatwe)** Wizualizuj proces forward: weź jeden obraz i wykreśl `x_t` dla `t in [0, 100, 250, 500, 750, 1000]`. Zweryfikuj, że `x_1000` wygląda jak czysty szum Gaussowski.
2. **(Średnie)** Trenuj TinyUNet na zbiorze danych synthetic-circles przez 20 epok i spróbkuj 16 kółek. Porównaj próbkowanie DDPM (1000 kroków) i DDIM (50 kroków) — czy produkują podobne obrazy z tego samego ziarna szumu?
3. **(Trudne)** Zaimplementuj cosinusowy harmonogram szumu (Nichol & Dhariwal, 2021): `alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi / 2)`. Trenuj ten sam model z harmonogramami liniowym i cosinusowym i pokaż, że cosinus daje lepsze próbki przy niskich liczbach kroków.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|----------------|--------------------------|
| Forward process | "Dodawaj szum w czasie" | Ustalanie łańcucha Markowa, który korumpuje obraz w szum Gaussowski przez T kroków |
| Reverse process | "Usuwaj szum krok po kroku" | Uczona dystrybucja, która wraca od szumu do obrazu |
| Epsilon prediction | "Przewiduj szum" | Cel treningowy: `epsilon_theta(x_t, t)` przewiduje szum dodany w kroku t |
| Beta schedule | "Ilości szumu" | Sekwencja T małych wariancji, które definiują ile szumu wchodzi na krok |
| alpha_bar_t | "Skumulowany współczynnik zachowania" | Iloczyn (1 - beta_s) do czasu t; większe t oznacza mniej sygnału |
| DDPM sampler | "Ancestralny, stochastyczny" | Próbkuje każde x_{t-1} z jego warunkowego rozkładu Gaussowskiego; 1000 kroków |
| DDIM sampler | "Deterministyczny, szybki" | Przepisuje próbkowanie jako deterministyczne ODE; 20-100 kroków z podobną jakością |
| Time conditioning | "Powiedz modelowi, które t" | Sinusoidalne embedding t wstrzykiwane w U-Net, żeby wiedział poziom szumu |

## Dalsza lektura

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) — artykuł, który uczynił dyfuzję praktyczną i pokonał GANy na FID
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672) — harmonogram cosinusowy i parametryzacja v
- [DDIM (Song, Meng, Ermon, 2020)](https://arxiv.org/abs/2010.02502) — deterministyczny sampler, który umożliwił inference w czasie rzeczywistym
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) — zunifikowany widok każdego wyboru projektowego dyfuzji; aktualna najlepsza referencja