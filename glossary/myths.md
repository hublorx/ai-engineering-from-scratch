# Mity o AI obalone

Powszechne nieporozumienia dotyczące AI, uczenia maszynowego i głębokiego uczenia. Każde wyjaśnione z tym, co faktycznie się dzieje.

---

## "AI rozumie język"

**Rzeczywistość:** Duże modele językowe przewidują następny token na podstawie statystycznych wzorców w danych treningowych. Nie mają rozumienia, przekonań, ani modelu świata (co możemy udowodnić). Są bardzo dobre w dopasowywaniu wzorców w miliardach przykładów. Wynik wygląda jak rozumienie, bo wzorce są wystarczająco bogate, żeby pokryć większość sytuacji.

**Dlaczego to ma znaczenie:** Jeśli traktujesz LLM jako silnik rozumowania, zaskoczy Cię, gdy pewnego razu powie coś z błędami. Jeśli traktujesz go jako dopasowywacz wzorców, zaprojektujesz lepsze systemy wokół niego.

---

## "Więcej parametrów = mądrzejszy model"

**Rzeczywistość:** Model z 7B parametrów wytrenowany na wysokiej jakości danych z dobrymi technikami może przewyższać model 70B wytrenowany na śmieciach. Chinchilla pokazała, że większość modeli była nadmiernie sparametryzowana i niedostatecznie trenowana. Jakość i ilość danych treningowych ma znaczenie tak samo duże jak rozmiar modelu. Phi-2 (2.7B) pokonał modele 10x większe na wielu benchmarkach.

**Dlaczego to ma znaczenie:** Nie wybieraj domyślnie największego modelu. Dobierz rozmiar modelu do zadania i budżetu.

---

## "Sieci neuronowe to czarne skrzynki"

**Rzeczywistość:** Mamy narzędzia do zrozumienia, czego uczą się sieci neuronowe. Wizualizacja attention pokazuje, na które tokeny model zwraca uwagę. Klasyfikatory probingowe ujawniają, jakie informacje są przechowywane w ukrytych reprezentacjach. Interpretowalność mechanistyczna znajduje rzeczywiste obwody (induction heads, detektory cech). To nie pełna przejrzystość, ale nie jest też czarną skrzynką.

**Dlaczego to ma znaczenie:** Możesz debugować sieci neuronowe. Analiza gradientów, wizualizacja aktywacji i mapy attention to realne narzędzia omawiane w tym kursie.

---

## "AI zastąpi programistów"

**Rzeczywistość:** AI zmieniło programowanie, ale go nie zastąpiło. AI pisze szablonowy kod. Ludzie projektują systemy, podejmują decyzje architektoniczne, przeglądają poprawność i obsługują przypadki, w których AI się myli. Rola przesunęła się od "pisz każdą linię" do "przeglądaj, kieruj i architektyzuj." Najlepsi inżynierowie używają AI jako narzędzia, nie boją się go jako zastępcę.

**Dlaczego to ma znaczenie:** Uczysz się inżynierii AI, czyli programowania + AI. Obie umiejętności razem są warte więcej niż każda z osobna.

---

## "Potrzebujesz doktoratu z matmy, żeby robić AI"

**Rzeczywistość:** Potrzebujesz matematyki ze szkoły średniej plus konkretnych tematów z Faz 1 tego kursu. Algebra liniowa, rachunek różniczkowy, prawdopodobieństwo i optymalizacja. Nie potrzebujesz dowodów. Potrzebujesz intuicji, co operacje robią i dlaczego mają znaczenie. Jeśli umiesz mnożyć macierze i obliczać pochodne, możesz budować sieci neuronowe.

**Dlaczego to ma znaczenie:** Faza 1 istnieje, żeby dać Ci dokładnie tę matematykę, której potrzebujesz, nic więcej.

---

## "GPT oznacza General Purpose Technology"

**Rzeczywistość:** GPT oznacza Generative Pre-trained Transformer. Generative = produkuje tekst. Pre-trained = trenowany raz na dużym korpusie przed adaptacją. Transformer = architektura z pracy "Attention Is All You Need" z 2017 roku.

---

## "Temperature sprawia, że AI jest bardziej kreatywna"

**Rzeczywistość:** Temperature skaluje logity przed softmax. Wyższa temperatura = bardziej płaska dystrybucja prawdopodobieństwa = bardziej losowy wybór tokenu. Niższa temperatura = ostrzejsza dystrybucja = bardziej deterministyczny. To nie kreatywność, to losowość. Model z wysoką temperaturą nie myśli intensywniej, po prostu rozważa mniej prawdopodobne tokeny.

**Dlaczego to ma znaczenie:** Gdy Twoje wyniki są zbyt powtarzalne, podnieś temperaturę. Gdy są zbyt chaotyczne, obniż ją. To pokrętło losowości, nic więcej.

---

## "Fine-tuning uczy model nowej wiedzy"

**Rzeczywistość:** Fine-tuning dostosowuje, jak model wykorzystuje istniejącą wiedzę, nie to, co wie. Jeśli informacja nie była w danych pre-treningu, fine-tuning jej nie doda wiarygodnie. Fine-tuning jest lepszy do zmiany zachowania (styl, format, ton, wzorce specyficzne dla zadania) niż do dodawania faktów. W przypadku nowej wiedzy użyj RAG.

**Dlaczego to ma znaczenie:** Jeśli model ma znać wewnętrzne dokumenty Twojej firmy, użyj RAG. Jeśli ma odpowiadać w konkretnym formacie, użyj fine-tuningu.

---

## "Większe okno kontekstowe = lepsze"

**Rzeczywistość:** Modele degradują na długich kontekstach. Problem "zagubiony w środku" oznacza, że modele zwracają więcej uwagi na początek i koniec długich promptów, a mniej na środek. Okno kontekstowe 200K nie oznacza, że model wykorzystuje wszystkie 200K tokenów równie dobrze. Ponadto dłuższe konteksty kosztują więcej i są wolniejsze.

**Dlaczego to ma znaczenie:** Nie wrzucaj wszystkiego do kontekstu. Bądź selektywny. RAG z ukierunkowanym wyszukiwaniem bije wrzucanie całego dokumentu.

---

## "Agenci AI są autonomiczni"

**Rzeczywistość:** Obecni agenci AI działają w pętli: pomyśl, działaj, obserwuj, powtórz. Podążają za wzorcem, który definiuje harness. Nie mają celów, planów, ani samoświadomości. To reaktywne systemy, które używają LLM do decydowania, które narzędzie wywołać następne. "Autonomia" pochodzi z pętli, nie z AI.

**Dlaczego to ma znaczenie:** Budując agentów, budujesz pętlę, narzędzia i mechanizmy zabezpieczające. LLM jest tylko komponentem decyzyjnym w Twoim systemie.

---

## "Transformery rozumieją kolejność dzięki positional encoding"

**Rzeczywistość:** Transformery nie mają naturalnego poczucia kolejności. Self-attention traktuje wejście jako zbiór, nie sekwencję. Positional encoding to hack do wstrzykiwania informacji o kolejności przez dodanie wektorów zależnych od pozycji do wejścia. Różne metody (sinusoidal, learned, RoPE, ALiBi) obsługują to inaczej. Żadna z nich tak naprawdę nie daje modelowi sekwencyjnego zrozumienia w sposób, w jaki miały to RNN-y.

**Dlaczego to ma znaczenie:** Dlatego badania nad positional encoding są wciąż aktywne. To wystarczająco rozwiązany problem dla większości zastosowań, ale fundamentalnie jest to obejście.

---

## "Pre-training to tylko czytanie internetu"

**Rzeczywistość:** Pre-training to przewidywanie następnego tokenu na ogromnym korpusie. Model uczy się przewidywać, co przyjdzie dalej, gdy dane jest to, co było przedtem. Poprzez ten prosty cel, uczy się gramatyki, faktów, wzorców rozumowania, struktury kodu i nie tylko. Ale uczy się też internetowego bełkotu, uprzedzeń i nieprawidłowych informacji. Kuration danych, filtrowanie i deduplikacja mają ogromne znaczenie.

**Dlaczego to ma znaczenie:** Śmieci na wejściu, śmieci na wyjściu. Jakość danych pre-treningu to jeden z największych wyróżników między modelami.

---

## "RLHF wyrównuje AI z ludzkimi wartościami"

**Rzeczywistość:** RLHF wyrównuje AI z preferencjami konkretnych ludzi, którzy dostarczyli feedback. Ci ludzie nie zgadzają się ze sobą, mają uprzedzenia i nie mogą pokryć każdej sytuacji. RLHF sprawia, że model jest pomocny i nieszkodliwy w sposobach zdefiniowanych przez oceniających, nie wyrównany z jakimkolwiek uniwersalnym systemem ludzkich wartości.

**Dlaczego to ma znaczenie:** RLHF to technika treningowa, nie rozwiązanie problemu alignment. To jedno narzędzie w większym zestawie.

---

## "Embeddings przechwytują znaczenie"

**Rzeczywistość:** Embeddings przechwytują wzorce współwystępowania statystycznego. Słowa, które pojawiają się w podobnych kontekstach, dostają podobne wektory. To koreluje ze znaczeniem wystarczająco dobrze, żeby być użytecznym, ale to nie jest semantyczne rozumienie. "King - Man + Woman = Queen" działa przez wzorce dystrybucyjne, nie dlatego, że model rozumie monarchię lub płeć.

**Dlaczego to ma znaczenie:** Embeddings są potężne do wyszukiwania podobieństw, klastrowania i retrieval. Ale nie interpretuj za bardzo, co "podobne" oznacza.

---

## "Zero-shot oznacza brak treningu"

**Rzeczywistość:** Zero-shot oznacza brak przykładów specyficznych dla zadania w czasie wnioskowania. Model był wciąż trenowany na miliardach tokenów. Po prostu nie widział przykładów tego konkretnego formatu zadania. Generalizuje z wzorców pre-treningu. Few-shot oznacza podanie kilku przykładów w prompcie. Żaden z nich nie oznacza, że model nauczył się bez treningu.

---

## "Modele AI uczą się jak ludzie"

**Rzeczywistość:** Ludzie uczą się z niewielu przykładów, generalizują między domenami i aktualizują przekonania ciągle. Sieci neuronowe potrzebują milionów przykładów, generalizują w ramach swojej dystrybucji treningowej i mają stałe wagi po treningu. Analogia do uczenia się jest luźna w najlepszym razie. Backpropagation nie ma nic wspólnego z tym, jak uczą się biologiczne neurony.

**Dlaczego to ma znaczenie:** Nie antropomorfizuj modeli. Prowadzi to do złych oczekiwań co do tego, co mogą i czego nie mogą robić.

---

## "Prawa skalowania oznaczają, że większy zawsze lepszy"

**Rzeczywistość:** Prawa skalowania opisują przewidywalne relacje między compute, danymi i rozmiarem modelu. Pokazują malejące korzyści: podwajanie parametrów nie podwaja wydajności. Zakładają też, że skalujesz dane proporcjonalnie. Wiele praktycznych usprawnień pochodzi z lepszych architektur, technik treningowych i jakości danych, nie tylko ze skali.

**Dlaczego to ma znaczenie:** Model 7B z dobrym inżynieringiem może rozwiązać Twój problem. Nie sięgaj domyślnie po 70B.

---

## "Open source AI to to samo co open weights"

**Rzeczywistość:** Większość modeli "open source" to open weights. Dostajesz pliki modelu, ale nie dane treningowe, kod treningowy ani pipeline danych. Prawdziwy open source (jak OLMo) udostępnia wszystko: dane, kod, pośrednie checkpointy, ewaluację. Open weights jest użyteczne, ale to nie to samo zobowiązanie co open source.

**Dlaczego to ma znaczenie:** Wiedź, co dostajesz. Open weights pozwala uruchamiać i fine-tune'ować. Prawdziwy open source pozwala odtwarzać i rozumieć.

---

## "Prompt engineering to nie prawdziwe inżynierowanie"

**Rzeczywistość:** Prompt engineering to projektowanie systemów. Projektujesz interfejs między ludzkim zamiarem a zachowaniem modelu. Dobry prompt engineering wymaga zrozumienia tokenizacji, wzorców attention, limitów okna kontekstowego i parsowania wyjścia. To bliższe projektowaniu API niż "rozmawianiu miło z AI."

**Dlaczego to ma znaczenie:** Ten kurs uczy prompt engineering jako realnej dyscypliny inżynieryjnej w Fazie 11.

---

## "CNN są przestarzałe, wszystko to teraz transformery"

**Rzeczywistość:** Vision Transformers (ViT) biją CNN na wielu benchmarkach, ale CNN są wciąż szeroko używane. Są szybsze do wnioskowania, dobrze działają na mobile/edge, potrzebują mniej danych i mają użyteczne inductive biases (niezmienniczość translacji, lokalne wzorce). Wiele produkcyjnych systemów wizyjnych wciąż używa CNN. Najlepsze architektury często łączą oba podejścia.

**Dlaczego to ma znaczenie:** Ucz się obu (Fazy 4 i 7). Używaj tego, co działa dla Twoich ograniczeń.

---

## "Potrzebujesz ogromnej mocy obliczeniowej, żeby trenować użyteczne modele"

**Rzeczywistość:** Potrzebujesz ogromnej mocy, żeby pre-trenować modele bazowe. Ale fine-tuning, LoRA i transfer learning pozwalają adaptować modele na jednym GPU. Wiele użytecznych aplikacji AI nie wymaga w ogóle treningu, tylko dobrego promptingu i RAG. "Bariera compute" dotyczy budowania modeli bazowych, nie używania ich.

**Dlaczego to ma znaczenie:** Możesz budować realne aplikacje AI z laptopem. Ten kurs to udowadnia.
