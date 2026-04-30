# Chatboty — Od regułowych do neuronowych do agentów LLM

> ELIZA odpowiadała przez dopasowywanie wzorców. DialogFlow mapował intencje. GPT odpowiadał z wag. Claude uruchamia narzędzia i weryfikuje. Każra era rozwiązywała najgorszy błąd poprzedniej.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 13 (Question Answering), Faza 5 · 14 (Information Retrieval)
**Szacowany czas:** ~75 minut

## Problem

Użytkownik mówi „Chcę zmienić mój lot." System musi ustalić, czego chce, jakich informacji brakuje, jak je zdobyć i jak ukończyć działanie. Potem użytkownik mówi „poczekaj, a co jeśli zamiast tego anuluję?" i system musi pamiętać kontekst, przełączyć zadanie i zachować stan.

Konwersacja jest trudna dla systemu ML. Wejście jest otwarte. Wyjście musi być spójne przez wiele tur. System może potrzebować działać na świecie (zmienić lot, obciążyć kartę). Każdy błędny krok jest widoczny dla użytkownika.

Architektury chatbotów przeszły przez cztery paradygmaty, z których każdy został wprowadzony, ponieważ poprzedni zawodził zbyt widocznie. Ta lekcja przechodzi je po kolei. Krajobraz produkcyjny 2026 to hybryda dwóch ostatnich.

## Koncepcja

![Ewolucja chatbotów: regułowe → retrieval → neuronowe → agent](../assets/chatbot.svg)

**Regułowe (ELIZA, AIML, DialogFlow).** Ręcznie tworzone wzorce dopasowują dane wejściowe użytkownika i produkują odpowiedzi. Klasyfikatory intencji kierują do predefiniowanych przepływów. Automaty stanów z wypełnianiem slotów zbierają wymagane informacje. Działa doskonale w wąskim zakresie, dla którego zostało zaprojektowane. Natychmiast zawodzi poza nim. Wciąż trafia do krytycznych domen bezpieczeństwa (uwierzytelnianie bankowe, rezerwacja linii lotniczych), gdzie halucynacje nie są tolerowane.

**Oparte na retrieval.** System typu FAQ. Koduje każdą parę (wypowiedź, odpowiedź). W czasie wykonywania koduje wiadomość użytkownika i pobiera najbliższą zapisaną odpowiedź. Pomyśl o klasycznej funkcji „podobnych artykułów" w Zendesk. Lepiej radzi sobie z parafrazami niż reguły. Brak generacji, więc brak halucynacji.

**Neuronowe (seq2seq).** Enkoder-dekoder trenowany na logach konwersacji. Generuje odpowiedzi od zera. Płynne, ale podatne na generyczne wyniki („Nie wiem") i dryf faktów. Nigdy niezawodnie na temat. Powód, dla którego Google, Facebook i Microsoft mieli rozczarowujące chatboty w 2016-2019.

**Agenci LLM.** Model językowy opakowany w pętlę, która planuje, wywołuje narzędzia i weryfikuje wyniki. To nie chatbot z długim promptem. To pętla agenta: plan → wywołaj narzędzie → obserwuj wynik → zdecyduj o następnym kroku. Retrieval-first grounding (RAG) zapobiega halucynacjom. Wywołania narzędzi pozwalają mu faktycznie wykonywać działania. To architektura 2026.

Cztery paradygmaty nie są sekwencyjnymi zastąpieniami. Chatbot produkcyjny 2026 kieruje przez wszystkie cztery: regułowe do uwierzytelniania i destrukcyjnych działań, retrieval do FAQ, generację neuronową do naturalnego formułowania, agenta LLM do niejasnych otwartych zapytań.

## Zbuduj to

### Krok 1: regułowe dopasowywanie wzorców

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA w 20 liniach. Trik z refleksją („I feel sad" → „Why do you feel sad") to kanoniczna demonstracja psychoterapeuty z Weizenbaum 1966. Wciąż instruktywna.

### Krok 2: oparte na retrieval (FAQ)

Ten ilustracyjny fragment wymaga `pip install sentence-transformers` (co ściąga torch). Uruchamialny `code/main.py` dla tej lekcji używa zamiast tego similarity Jaccard ze stdlib, więc lekcja działa bez zewnętrznych zależności.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

Odcięcie na podstawie progu to kluczowy wybór projektowy. Jeśli najlepsze dopasowanie nie jest wystarczająco bliskie, zwróć `None` i pozwól systemowi eskalować.

### Krok 3: generacja neuronowa (baseline)

Użyj małego instrukcjowo dostrojonego enkodera-dekodera (FLAN-T5) lub dostrojonego modelu konwersacyjnego. W 2026 nie nadaje się do użytku produkcyjnego samodzielnie (sprzeczności, dryf poza temat, faktyczny bezsens), ale trafia do hybrydowych systemów do naturalnego formułowania. Modele decoder-only w stylu DialoGPT wymagają jawnych separatorów tur i obsługi EOS do produkcji spójnych odpowiedzi; pipeline text2text FLAN-T5 działa out of the box jako przykład dydaktyczny.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Krok 4: pętla agenta LLM

Produkcyjny kształt 2026:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            result = tools[tool_name](**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

Trzy rzeczy do nazwania. Narzędzia to wywoływalne funkcje, które LLM może wywołać. Pętla kończy się, gdy LLM zwraca ostateczną odpowiedź zamiast wywołania narzędzia. Budżet kroków zapobiega nieskończonym pętlom przy niejasnych zadaniach.

Prawdziwa produkcja dodaje: retrieval-first grounding (wstrzykiwanie odpowiednich dokumentów przed każdym wywołaniem LLM), guardrails (odmowa destrukcyjnych działań bez potwierdzenia), observability (logowanie każdego kroku) i ewaluacje (zautomatyzowane sprawdzanie, że zachowanie agenta pozostaje zgodne ze specyfikacją).

### Krok 5: hybrydowe routingowanie

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

Wzorzec: deterministyczne reguły dla wszystkiego destrukcyjnego, retrieval dla gotowych FAQ, agenci LLM dla wszystkiego innego. To jest to, co trafia do produkcyjnych systemów obsługi klienta 2026.

## Użyj tego

Stack 2026:

| Przypadek użycia | Architektura |
|-----------------|--------------|
| Rezerwacja, płatność, uwierzytelnianie | Regułowe automaty stanów + wypełnianie slotów |
| FAQ obsługi klienta | Retrieval na wyselekcjonowanych odpowiedziach |
| Otwarta pomoc czatowa | Agent LLM z RAG + wywołaniami narzędzi |
| Wewnętrzne narzędzia / asystenci IDE | Agent LLM z wywołaniami narzędzi (wyszukiwanie, czytanie, pisanie) |
| Chatboty towarzyszące / postaciowe | Dostrojony LLM z systemowym promptem persony, retrieval na wiedzy |

Zawsze używaj hybrydowego routingowania w produkcji. Żadna pojedyncza architektura nie obsługuje dobrze każdego żądania. Sama warstwa routingowania to typowo mały klasyfikator intencji.

## Tryby awarii, które wciąż trafiają do produkcji

- **Pewna fabrykacja.** Agent LLM twierdzi, że ukończył działanie, którego nie wykonał. Łagodzenie: weryfikuj wyniki, loguj wywołania narzędzi, nigdy nie pozwalaj LLM twierdzić, że coś zrobił bez pomyślnego zwrotu z narzędzia.
- **Prompt injection.** Użytkownik wstawia tekst, który nadpisuje system prompt. Sklasyfikowany LLM01 w OWASP Top 10 for LLM Applications 2025. Dwa smaki: bezpośrednia iniekcja (wklejona do czatu) i pośrednia iniekcja (ukryta w dokumentach, e-mailach lub wynikach narzędzi, które agent czyta).

  Wskaźniki ataków różnią się w zależności od scenariusza. Zmierzone wskaźniki sukcesu wahają się od ~0,5-8,5% wśród frontier models w ogólnych benchmarkach użycia narzędzi i kodowania. Konkretne wysokiego ryzyka konfiguracje (adaptacyjne ataki na AI coding agents, podatna orkiestracja) osiągnęły ~84%. Produkcyjne CVE obejmują EchoLeak (CVE-2025-32711, CVSS 9.3) — lukę w zabezpieczeniach typu zero-click umożliwiającą eksfiltrację danych w Microsoft 365 Copilot wyzwalaną przez e-mail kontrolowany przez atakującego.

  Łagodzenia: traktuj dane wejściowe użytkownika jako niezaufane przez całą pętlę; sanityzuj przed wywołaniami narzędzi; izoluj wyniki narzędzi od głównego prompta; używaj wzorca Plan-Verify-Execute (PVE), gdzie agent najpierw planuje, następnie weryfikuje każde działanie względem tego planu przed wykonaniem (to powstrzymuje wyniki narzędzi przed wstrzykiwaniem nowych nieplanowanych działań); wymagaj potwierdzenia użytkownika dla destrukcyjnych działań; stosuj least-privilege do zakresów narzędzi.

  Żadna ilość prompt engineeringu nie eliminuje całkowicie tego ryzyka. Zewnętrzne warstwy obrony runtime (LLM Guard, walidacja allowlist, wykrywanie semantycznych anomalii) są wymagane.
- **Rozrost zakresu.** Agent schodzi z zadania, bo wywołanie narzędzia zwróciło pobocznie powiązane informacje. Łagodzenie: zawężaj kontrakty narzędzi; utrzymuj system prompt skoncentrowany; dodawaj ewaluacje na wskaźnik zejścia z tematu.
- **Nieskończone pętle.** Agent ciągle wywołuje to samo narzędzie. Łagodzenie: budżet kroków, deduplikacja wywołań narzędzi, sędzia LLM na „czy robimy postępy."
- **Wycieńczenie okna kontekstowego.** Długie rozmowy wypychają najwcześniejsze tury poza kontekst. Łagodzenie: podsumowuj starsze tury, pobieraj pasujące przeszłe tury przez podobieństwo lub używaj long-context model.

## Wyślij to

Zapisz jako `outputs/skill-chatbot-architect.md`:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj regułową odpowiedź powyżej z 10 wzorcami dla chatbota zamawiania w kawiarni. Przetestuj przypadki brzegowe: podwójne zamówienia, modyfikacje, anulowanie, niejasna intencja.
2. **Średnie.** Zbuduj hybrydowe FAQ + fallback LLM. 50 puszkowych wpisów FAQ dla produktu SaaS, fallback LLM z retrieval na stronie dokumentacji. Zmierz wskaźnik odmów i dokładność na 100 prawdziwych pytaniach wsparcia.
3. **Trudne.** Zaimplementuj pętlę agenta powyżej z trzema narzędziami (search, read-user-data, send-email). Uruchom ewaluację z 50 scenariuszami testowymi w tym próbami prompt injection. Raportuj wskaźnik zejścia z tematu, wskaźnik nieudanych zadań i sukces jakiejkolwiek iniekcji.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-------------------------|
| Intencja | Czego użytkownik chce | Etykieta kategorialna (book_flight, reset_password). Kierowana do handlera. |
| Slot | Kawałek informacji | Parametr, którego bot potrzebuje (data, destynacja). Wypełnianie slotów to sekwencja zapytań. |
| RAG | Retrieval plus generation | Pobierz odpowiednie dokumenty, następnie uziemiń odpowiedź LLM. |
| Tool call | Wywołanie funkcji | LLM emituje ustrukturyzowane wywołanie z nazwą + argumentami. Runtime wykonuje, zwraca wynik. |
| Agent loop | Planuj, działaj, weryfikuj | Kontroler, który uruchamia wywołania LLM przeplatane wywołaniami narzędzi aż do ukończenia zadania. |
| Prompt injection | Użytkownik atakuje prompt | Złośliwy input, który próbuje nadpisać system prompt. |

## Dalsza lektura

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — oryginalny regułowy artykuł o chatbotach.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — późny neuronowy artykuł o chatbotach Google, tuż przed przejęciem przez agentów LLM.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — artykuł, który nazwał wzorzec pętli agenta.
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — produkcyjne wskazówki z 2024, które wciąż obowiązują w 2026.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — artykuł o prompt injection.
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — ranking, który uczynił prompt injection największym problemem bezpieczeństwa.
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — praktyczna obrona warstwy orkiestracji, w tym Plan-Verify-Execute i przepływy potwierdzenia użytkownika.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — kanoniczne CVE typu zero-click eksfiltracji danych z pośredniego prompt injection. Przypadek referencyjny, dlaczego agenci z dostępem do zapisu potrzebują obrony runtime.