# Multi-Agent RL

> Single-agent RL zakłada, że środowisko jest stacjonarne. Umieść dwóch uczących się agentów w tym samym świecie i to założenie się łamie: każdy agent jest częścią środowiska drugiego, a oboje się zmieniają. Multi-agent RL to zbiór sztuczek, dzięki którym uczenie zbiega się, gdy założenie Markowa nie jest już spełnione.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 9 · 04 (Q-learning), Faza 9 · 06 (REINFORCE), Faza 9 · 07 (Actor-Critic)
**Szacowany czas:** ~45 minut

## Problem

Robot uczący się nawigacji w pokoju to problem RL z jednym agentem. Drużyna soccerowa nie jest. AlphaStar kontra przeciwnicy w StarCraft nie jest. Rynek licytujących agentów nie jest. Dwa samochody negocjujące skrzyżowanie nie jest. Wiele realnych problemów wiele-do-wielu nie jest.

W każdym ustawieniu multi-agent, z perspektywy jednego agenta, inni agenci *są* częścią środowiska. Gdy się uczą i zmieniają swoje zachowanie, środowisko staje się niestacjonarne. Własność Markova — „następny stan zależy tylko od bieżącego stanu i mojej akcji" — zostaje naruszona, ponieważ następny stan zależy również od tego, co *inni agenci* wybrali, a ich polityki są zmiennymi celami.

To łamie dowody zbieżności dla metod tablicowych (gwarancja Q-learningu zakłada stacjonarne środowisko). To łamie również naiwne deep RL: agenci gonią się w pętlach, nigdy nie zbiegają do stabilnej polityki. Potrzebujesz technik specyficznych dla multi-agent: scentralizowanego treningu / zdecentralizowanego wykonania, kontrfaktycznych baz, ligi, self-play.

Aplikacje 2026: roje robotów, routing ruchu, floty pojazdów autonomicznych, symulatory rynku, systemy LLM multi-agent (Faza 16), każda gra z więcej niż jednym inteligentnym graczem.

## Koncepcja

![Cztery reżimy MARL: indep, centralized critic, self-play, league](../assets/marl.svg)

**Formalizm: Markov Game.** Uogólnienie MDP: stany `S`, wspólna akcja `a = (a_1, …, a_n)`, przejście `P(s' | s, a)`, oraz nagrody per-agent `R_i(s, a, s')`. Każdy agent `i` maksymalizuje własny zwrot pod własną polityką `π_i`. Jeśli nagrody są identyczne, jest to **w pełni kooperacyjne**. Jeśli zero-sum, jest to **adwersarialne**. Jeśli mieszane, jest to **ogólna suma**.

**Główne wyzwania:**

- **Niestacjonarność.** `P(s' | s, a_i)` z perspektywy agenta `i` zależy od `π_{-i}`, który się zmienia.
- **Przydział zasług (credit assignment).** Przy wspólnej nagrodzie, który agent ją spowodował?
- **Koordynacja eksploracji.** Agenci muszą eksplorować komplementarne strategie, a nie redundancko eksplorować ten sam stan.
- **Skalowalność.** Wspólna przestrzeń akcji rośnie wykładniczo w `n`.
- **Częściowa obserwowalność.** Każdy agent widzi tylko własną obserwację; globalny stan jest ukryty.

**Cztery dominujące reżimy:**

**1. Independent Q-learning / independent PPO (IQL, IPPO).** Każdy agent uczy się własnego Q lub polityki, traktując innych jako część środowiska. Proste, czasami działa (szczególnie z experience replay działającym jako sztuczka wygładzania modelu agenta). Teoretyczna zbieżność: brak. W praktyce: OK dla luźno sprzężonych zadań, złe dla ściśle sprzężonych.

**2. Centralized training, decentralized execution (CTDE).** Najczęstszy nowoczesny paradygmat. Każdy agent ma własną *politykę* `π_i`, która warunkuje na lokalnej obserwacji `o_i` — standardowe zdecentralizowane wykonanie przy wdrożeniu. Podczas *treningu*, scentralizowany krytyk `Q(s, a_1, …, a_n)` warunkuje na pełnym stanie globalnym i wspólnej akcji. Przykłady:
- **MADDPG** (Lowe et al. 2017): DDPG ze scentralizowanym krytykiem per agent.
- **COMA** (Foerster et al. 2017): kontrfaktyczna baza — pytaj „jaka byłaby moja nagroda, gdybym wykonał akcję `a'`?" — izoluje mój wkład.
- **MAPPO** / **IPPO** ze współdzielonym krytykiem (Yu et al. 2022): PPO ze scentralizowaną funkcją wartości. Dominujące w 2026 dla kooperacyjnego MARL.
- **QMIX** (Rashid et al. 2018): dekompozycja wartości — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` z monotonicznym miksowaniem.

**3. Self-play.** Dwie kopie tego samego agenta grają ze sobą. Polityka przeciwnika *to* moja polityka ze starego snapshotu. AlphaGo / AlphaZero / MuZero. OpenAI Five. Najlepiej działa dla gier zero-sum; sygnał treningowy jest symetryczny.

**4. League play.** Rozszerzenie self-play na środowiska ogólna-suma / adwersarialne: utrzymuj populację przeszłych i obecnych polityk, próbkuj przeciwnika z ligi, trenuj przeciwko niemu. Dodaje exploiterów (specjalizują się w pokonywaniu aktualnego najlepszego) i main exploiterów (specjalizują się w pokonywaniu exploiterów). AlphaStar (StarCraft II). Potrzebne, gdy gra dopuszcza cykle strategii typu kamień-nożyce-papier.

**Komunikacja.** Pozwól agentom wysyłać nauczone wiadomości `m_i` do siebie. Działa w ustawieniach kooperacyjnych. Foerster et al. (2016) pokazali, że różniczkowalna komunikacja między agentami może być trenowana end-to-end. Dzisiejsze systemy multi-agent oparte na LLM (Faza 16) zasadniczo komunikują się w języku naturalnym.

## Zbuduj To

Ta lekcja używa GridWorld 6×6 z dwoma kooperacyjnymi agentami. Zaczynają w przeciwnych rogach i muszą dotrzeć do wspólnego celu. Wspólna nagroda: `-1` za krok, gdy którykolwiek agent wciąż się porusza, `+10` gdy oboje dotrą. Zobacz `code/main.py`.

### Krok 1: środowisko multi-agent

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # dwoje agentów

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*Wspólna* przestrzeń akcji to `|A|² = 16`. Globalny stan to dwie pozycje.

### Krok 2: independent Q-learning

Każdy agent uruchamia własną tablicę Q kluczowaną na wspólnym stanie. W każdym kroku: oboje wybierają akcje ε-greedy, zbierają wspólne przejście, każdy aktualizuje własne Q wspólną nagrodą.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Działa na tym zadaniu, bo nagrody są gęste i wyrównane. Ponieka na ściśle sprzężonych zadaniach (np. gdzie jeden agent musi *czekać* na drugiego).

### Krok 3: scentralizowane Q z dekompozycją wartości

Użyj jednego Q dla wspólnych akcji `Q(s, a_1, a_2)`. Aktualizuj ze wspólnej nagrody. Zdecentralizuj przy wykonaniu przez marginalizację: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Handelje wykładniczą przestrzenią wspólnych akcji za *poprawny* globalny widok.

### Krok 4: prosty self-play (adwersarialny 2-agent)

Ten sam agent, dwie role. Trenuj agenta A przeciwko agentowi B; po `K` epizodach kopiuj wagi A do B. Symetryczny trening, konsekwentny postęp. Przepis AlphaZero w miniaturze.

## Pułapki

- **Niestacjonarne replay.** Experience replay z niezależnymi agentami jest gorsze niż single-agent, bo stare przejścia były generowane przez już nieaktualnych przeciwników. Fix: relabel lub wagowanie przez świeżość.
- **Niejednoznaczność przydziału zasług.** Wspólna nagroda po długim epizodzie; brak jasnego sposobu powiedzenia, który agent się przyczynił. Fix: kontrfaktyczne bazy (COMA), lub reward shaping per agent.
- **Dryf polityki / gonitwa.** Najlepsza odpowiedź każdego agenta zmienia się przy każdej aktualizacji drugiego. Fix: scentralizowany krytyk, wolne learning rates, lub zamrażanie jednego naraz.
- **Hackowanie nagród przez koordynację.** Agenci znajdują skoordynowane exploity, których projektant nie przewidział. Agenci aukcyjni zbiegają się do licytowania zero. Fix: staranne projektowanie nagród, ograniczenia behawioralne.
- **Redundancja eksploracji.** Obaj agenci eksplorują te same pary stan-akcja. Fix: entropy bonusy per-agent, lub warunkowanie roli.
- **Cykle ligowe.** Czysty self-play może utknąć w cyklu dominacji. Fix: league play z różnorodnymi przeciwnikami.
- **Eksplozja próbek.** `n` agentów × przestrzeń stanów × wspólne akcje. Aproksymuj przez aproksymację funkcji; rozkładane przestrzenie akcji (jedna głowa wyjściowa polityki per agent).

## Użyj To

Mapa aplikacji MARL 2026:

| Domena | Metoda | Uwagi |
|--------|--------|-------|
| Cooperative navigation / manipulation | MAPPO / QMIX | CTDE; współdzielony krytyk + zdecentralizowani aktorzy. |
| Gry dwuosobowe (szachy, Go, poker) | Self-play z MCTS (AlphaZero) | Zero-sum; symetryczny trening. |
| Złożone wieloosobowe (Dota, StarCraft) | League play + pretraining przez imitację | OpenAI Five, AlphaStar. |
| Floty pojazdów autonomicznych | CTDE MAPPO / PPO z attention | Częściowa obserwowalność; zmienne rozmiaru drużyn. |
| Rynki aukcyjne | Równowaga teorii gier + RL | Mean-field RL gdy `n` → ∞. |
| Systemy LLM multi-agent (Faza 16) | Komunikacja w języku naturalnym + warunkowanie roli | Pętla RL na warstwie planowania agentów. |

W 2026 największym obszarem wzrostu MARL są systemy oparte na LLM: roje agentów opartych na modelach językowych negocjujących, debatujących, budujących oprogramowanie. RL pojawia się jako optymalizacja preferencji na *poziomie trajektorii*, nie tokenów (Faza 16 · 03).

## Wyślij To

Zapisz jako `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Wybierz właściwy reżim multi-agent RL (IPPO, CTDE, self-play, league) dla danego zadania.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Ćwiczenia

1. **Łatwe.** Trenuj independent Q-learning na 2-agent cooperative GridWorld. Ile epizodów do średniego zwrotu > 0? Wykreśl krzywą wspólnego uczenia.
2. **Średnie.** Dodaj zadanie „koordynacji": cel jest osiągnięty tylko gdy oboje agenci wejdą na niego w tym samym ruchu. Czy independent Q nadal zbiega się? Co się psuje?
3. **Trudne.** Zaimplementuj scentralizowany krytyk dla treningu typu MAPPO i porównaj szybkość zbieżności z independent PPO na zadaniu koordynacji.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| Markov game | „Multi-agent MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; każdy agent ma własną nagrodę. |
| CTDE | „Centralized training, decentralized execution" | Wspólny krytyk w czasie treningu; polityka każdego agenta używa tylko lokalnej obs. |
| IPPO | „Independent PPO" | Każdy agent uruchamia PPO osobno. Prosty baseline; często niedoceniany. |
| MAPPO | „Multi-agent PPO" | PPO ze scentralizowaną funkcją wartości warunkującą na stanie globalnym. |
| QMIX | „Monotonic value decomposition" | `Q_tot = f_monotone(Q_1, …, Q_n)` pozwala na zdecentralizowany argmax. |
| COMA | „Counterfactual multi-agent" | Advantage = moje Q minus oczekiwane Q marginalizujące po mojej akcji. |
| Self-play | „Agent vs przeszłe ja" | Pojedynczy agent, dwie role; standard dla gier zero-sum. |
| League play | „Trening populacyjny" | Cache przeszłych polityk, próbkuj przeciwników z puli; obsługuje cykle strategii. |

## Dalsza Lecja

- Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG) — CTDE ze scentralizowanym krytykiem.
- Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA) — kontrfaktyczne bazy dla przydziału zasług.
- Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation — dekompozycja wartości z monotonicznością.
- Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO) — PPO jest zaskakująco silny dla MARL.
- Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar) — league play na skali.
- Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero) — czysty self-play w grach zero-sum.
- Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers — obejmuje krótkie omówienie ustawień multi-agent i problemu niestacjonarności, które CTDE jest zaprojektowane rozwiązać.
- Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview — przegląd obejmujący kooperacyjne, konkurencyjne i mieszane MARL z wynikami zbieżności.