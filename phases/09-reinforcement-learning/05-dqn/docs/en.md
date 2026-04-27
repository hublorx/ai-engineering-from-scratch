# Deep Q-Networks (DQN)

> 2013: Mnih wytrenował jedną sieć Q-learning na surowych pikselach, pokonał każdego klasycznego agenta RL na siedmiu grach Atari. 2015: rozszerzono do 49 gier, opublikowano w Nature, zapoczątkowano erę deep-RL. DQN to Q-learning plus trzy triki, które stabilizują aproksymację funkcji.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Czas:** ~75 minut

## Problem

Tabular Q-learning potrzebuje osobnej Q-wartości dla każdej pary (stan, akcja). Szachownica ma ~10⁴³ stanów. Klatka Atari to 210×160×3 = 100 800 cech. Tabular RL umiera przy tysiącach stanów, nie mówiąc o miliardach.

Poprawka jest oczywista wstecz: zastąp tablicę Q siecią neuronową, `Q(s, a; θ)`. Ale to, co oczywiste wstecz, zajęło dziesięciolecia. Naiwna aproksymacja funkcji z Q-learning rozbiega się w obliczu "śmiertelnej triady" — aproksymacja funkcji + bootstrapping + off-policy learning. Mnih et al. (2013, 2015) zidentyfikowali trzy triki inżynieryjne, które stabilizują uczenie:

1. **Experience replay** dekoreluje przejścia.
2. **Target network** zamraża cel bootstrapowy.
3. **Reward clipping** normalizuje wielkości gradientów.

DQN na Atari był pierwszym razem, gdy pojedyncza architektura z jednym zestawem hiperparametrów rozwiązała dziesiątki problemów sterowania z surowych pikseli. Wszystko, co "deep-RL" zbudowało od tego czasu — DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 — jest nałożone na tę bazę z trzema trikami.

## Koncepcja

![Pętla treningowa DQN: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**Cel.** DQN minimalizuje jednokrokowy TD loss na neuronowej funkcji Q:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = online network, aktualizowana co krok przez gradient descent. `θ^-` = target network, okresowo kopiowana z `θ` (co ~10 000 kroków). `D` = replay buffer przeszłych przejść.

**Trzy triki, w kolejności ważności:**

**Experience replay.** Ring buffer `~10⁶` przejść. Każdy krok treningowy próbkuje minibatch losowo równomiernie. To łamie korelację czasową (kolejne klatki są prawie identyczne), pozwala sieci uczyć się od rzadkich nagradzających przejść wielokrotnie i dekoleruje kolejne aktualizacje gradientów. Bez niego on-policy TD z siecią neuronową rozbiega się na Atari.

**Target network.** Używanie tej samej sieci `Q(·; θ)` po obu stronach równania Bellmana sprawia, że cel porusza się z każdą aktualizacją — "gonisz własny ogon." Poprawka: trzymaj drugą sieć `Q(·; θ^-)` z zamrożonymi wagami. Co `C` kroków kopiuj `θ → θ^-`. To stabilizuje cel regresji dla tysięcy kroków gradientowych naraz. Miękkie aktualizacje `θ^- ← τ θ + (1-τ) θ^-` (używane w DDPG, SAC) to gładsza odmiana.

**Reward clipping.** Wielkości nagród Atari wahają się od 1 do 1000+. Obcięcie do `{-1, 0, +1}` powstrzymuje dowolną grę od dominowania gradientu. Błędne gdy wielkość nagrody ma znaczenie; w porządku dla Atari, gdzie liczy się tylko znak.

**Double DQN.** Hasselt (2016) naprawia bias maksymalizacji: użyj online net do *wyboru* akcji, target net do jej *ewaluacji*.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

Zastąpienie drop-in, konsekwentnie lepsze. Używaj domyślnie.

**Inne usprawnienia (Rainbow, 2017):** prioritized replay (próbkuj przejścia o wysokim TD-error częściej), architektura dueling (osobne głowy `V(s)` i advantage), noisy networks (nauczone eksploracja), n-step returns, distributional Q (C51/QR-DQN), multi-step bootstrapping. Każde dodaje kilka procent; zyski są w przybliżeniu addytywne.

## Zbuduj to

Kod tutaj jest wyłącznie stdlib bez numpy-free — używamy ręcznie napisanej sieci MLP z jedną ukrytą warstwą na małym ciągłym GridWorld, więc każdy krok treningowy działa w mikrosekundach. Algorytm jest identyczny z DQN na Atari w skali.

### Krok 1: replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

~50 000 pojemności dla Atari; 5 000 wystarczy dla naszego środowiska zabawki.

### Krok 2: malutka sieć Q (manual MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Forward pass: linear → ReLU → linear. To jest cała sieć.

### Krok 3: aktualizacja DQN

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

 Kształt to Q-learning z Lesson 04 z dwiema różnicami: (a) backpropagujemy przez różniczkowalną `Q(·; θ)` zamiast indeksować tablicę, (b) cel używa `Q(·; θ^-)`.

### Krok 4: zewnętrzna pętla

Dla każdego epizodu, działaj ε-greedy na `Q(·; θ)`, pushuj przejścia do buffora, próbkuj minibatch, wykonaj krok gradientowy, okresowo sync `θ^- ← θ`. Wzorzec:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

Na naszym małym GridWorld z 16-wymiarowym stanem one-hot, agent uczy się niemal-optymalnej polityki w ~500 epizodach. Na Atari, skaluj to do 200M klatek i dodaj ekstraktor cech CNN.

## Pułapki

- **Śmiertelna triada.** Aproksymacja funkcji + off-policy + bootstrapping może się rozejść. DQN łagodzi to target net + replay; nie usuwaj żadnego z nich.
- **Eksploracja.** ε musi opadać, typowo od 1.0 do 0.01 przez pierwsze ~10% treningu. Bez wystarczającej wczesnej eksploracji Q-net zbiega się do lokalnego basenu.
- **Nadestymacja.** `max` nad zaszumionym Q jest biasowane w górę. Zawsze używaj Double DQN w produkcji.
- **Skala nagród.** Obcinaj lub normalizuj nagrody; wielkość gradientu jest proporcjonalna do wielkości nagrody.
- **Coldstart buffora replay.** Nie trenuj dopóki buffer nie ma kilku tysięcy przejść. Wczesne gradienty na ~20 próbkach przeuczają się.
- **Częstotliwość sync target.** Zbyt częsta ≈ brak target net; zbyt rzadka ≈ zgniłe cele. Atari DQN używa 10 000 kroków env. Zasada kciuka: sync co ~1/100 horyzontu treningowego.
- **Preprocessing obserwacji.** Atari DQN stackuje 4 klatki żeby uczynić stan Markowa. Każde env z info o prędkości potrzebuje frame-stackingu lub rekurencyjnego stanu.

## Użyj tego

W 2026 DQN rzadko jest state-of-the-art, ale pozostaje referencyjnym algorytmem off-policy:

| Zadanie | Metoda z wyboru | Dlaczego nie DQN? |
|---------|------------------|-------------------|
| Dyskretne akcje Atari-like | Rainbow DQN lub Muesli | Ten sam framework, więcej tricków. |
| Sterowanie ciągłe | SAC / TD3 (Phase 9 · 07) | DQN nie ma sieci polityki. |
| On-policy / wysoka przepustowość | PPO (Phase 9 · 08) | Brak replay buffer; łatwiej skalować. |
| Offline RL | CQL / IQL / Decision Transformer | Conservative Q targets, brak bootstrap blowups. |
| Duże przestrzenie dyskretnych akcji (rekomendacje) | DQN z action embedding, lub IMPALA | W porządku; dekoracja ma znaczenie. |
| LLM RL | PPO / GRPO | Poziom sekwencyjny, nie krokowy; inny loss. |

Lekcje nadal podróżują. Replay i target networks pojawiają się w SAC, TD3, DDPG, SAC-X, buforze self-play AlphaZero i każdej metodzie offline RL. Reward clipping żyje dalej jako normalizacja advantage w PPO. Architektura jest blueprintem.

## Wyślij to

Zapisz jako `outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Narysuj krzywą returnu per-epizod. Ile epizodów dopóki średnia krocząca nie przekroczy -10?
2. **Średnie.** Wyłącz target network (użyj online net dla obu stron celu Bellmana). Zmierz niestabilność treningu — czy return oscyluje lub się rozbiega?
3. **Trudne.** Dodaj Double DQN: użyj online net do wybrania `argmax a'`, target net do ewaluacji. Porównaj bias `Q(s_0, best_a)` vs prawdziwe `V*(s_0)` po 1,000 epizodów z vs bez Double DQN na zaszumionym GridWorld.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| DQN | "Deep Q-learning" | Q-learning z neuronową funkcją Q, replay buffer i target network. |
| Experience replay | "Shuffled transitions" | Ring buffer próbkowany równomiernie każdego kroku gradientowego; dekoleruje dane. |
| Target network | "Frozen bootstrap" | Okresowa kopia Q używana w celu Bellmana; stabilizuje trening. |
| Deadly triad | "Why RL diverges" | Aproksymacja funkcji + bootstrapping + off-policy = brak gwarancji zbieżności. |
| Double DQN | "Fix for maximization bias" | Online net wybiera akcję, target net ją ewaluuje. |
| Dueling DQN | "V and A heads" | Dekomponuj Q = V + A - mean(A); ten sam output, lepszy przepływ gradientu. |
| Rainbow | "All the tricks" | DDQN + PER + dueling + n-step + noisy + distributional w jednym. |
| PER | "Prioritized Replay" | Próbkuj przejścia proporcjonalnie do wielkości TD-error. |

## Dalsze czytanie

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — artykuł z warsztatów NeurIPS 2013, który zapoczątkował deep RL.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — artykuł Nature, DQN na 49 gier.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — artykuł o składaniu tricków.
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — jasne nowoczesne przedstawienie.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — podręcznikowe omówienie "śmiertelnej triady" (aproksymacja funkcji + bootstrapping + off-policy), którą target network i replay buffer DQNa mają łagodzić.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — referencyjny single-file DQN używany w studiach ablacyjnych; dobry do przeczytania obok wersji from-scratch z tej lekcji.