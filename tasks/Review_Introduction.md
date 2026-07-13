* Check paper (Pascual-Leone et al., 2005b).
* canonical instance sounds weird.
* "and adds complexity only when the added complexity earns its keep" sounds weird (earns its keep)

* "Each node’s activation function becomes an evolvable trait, so a population can discover that one neuron should saturate like tanh, another should rectify like ReLU, and a third should oscillate like a sine." rectify like sounds werid. 

* "its incident connections" sounds weird and we never have motivated what historical markers are.

* "on two structurally asymmetric reinforcement-learning environments, Hopper (11 observations, 3 actions) and Walker2D (17 observations, 6 actions)" Those environments are not reinforcement learning environments but physics simulators
* "If that coupling matters anywhere, it should matter where one network has to do several jobs at once. Multi-task reinforcement learning is the natural test. Consider Hopper and Walker2D: the same physics simulator, but different bodies, different observation sizes, different action spaces, and different ways of falling over." ~ There is no argument why it should matter in a multi-task setting. Also it is not Multi-task reinforcement learning

* "Once a single shared genome has to control both, the assumption that one nonlinearity covers every useful function shape stops being a convenience and becomes an empirical claim." This sounds like AI

* "The mechanical question of how to push a Hopper observation and a Walker2D observation through the same network is usually glossed over or solved with custom architectures." ~ mechanical question sounds like ai. Also we should rather say how to generally push two different observations with different sizes through. 

* "empirically. Fitness aggregation, the way two different reward signals are collapsed into one selection pressure, receives even less attention, despite being the single design choice with the most influence over what the population learns." is this a true statement? NSGA II for example is used 



* "Formally, for a network f : R17→ R6, the Hopper action is f([oHopper; 06])1:3 and the Walker2D action is f(oWalker)1:6." Lets provide those two as bullet points anstatt im Fließtext
* " " ~ this sounds weird and like AI
* "The scheme is deliberately small, and that is the point: it is the least machinery under which we can ask whether one genome generalises across two bodies without the architecture arranging the answer in advance." ~ This sounds like AI
* "We chose the palette for complementary roles, keeping it small: boundedsaturating units (tanh, σ) for gating and balance, ReLU for sparse unbounded propagation, sine for the oscillations that rhythmic locomotion invites, and identity for proportional pass-through paths." ~ This reasoning sounds wrong and AI-ish


* We do not use machinery as a word. It sounds artificial and wrong
* "Here the coupling between activation diversityand speciation stops being a design detail and becomes something the experiments can test." ~ This sounds weird 
* 