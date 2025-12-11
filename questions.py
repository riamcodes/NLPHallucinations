from pathlib import Path
import json
from typing import Any, Dict, List

OUTPUTS_DIR = Path("outputs")

SIMPLE_FACTUAL_QUESTIONS = [
    {
        "qid": "SF001",
        "category": "simple_factual",
        "question": "What is the capital of France?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Straightforward factual question with a clear, verifiable answer"
        }
    },
    {
        "qid": "SF002",
        "category": "simple_factual",
        "question": "Who wrote Romeo and Juliet?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Well-known historical fact about a famous playwright"
        }
    },
    {
        "qid": "SF003",
        "category": "simple_factual",
        "question": "What is the largest planet in our solar system?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Astronomical fact with a definitive answer"
        }
    },
    {
        "qid": "SF004",
        "category": "simple_factual",
        "question": "When was the United States founded?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Clear historical date with verified documentation"
        }
    },
    {
        "qid": "SF005",
        "category": "simple_factual",
        "question": "What is the chemical symbol for gold?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Straightforward scientific fact"
        }
    },
    {
        "qid": "SF006",
        "category": "simple_factual",
        "question": "Who painted the Mona Lisa?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Well-known historical art fact"
        }
    },
    {
        "qid": "SF007",
        "category": "simple_factual",
        "question": "What is the speed of light?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Precise scientific constant"
        }
    },
    {
        "qid": "SF008",
        "category": "simple_factual",
        "question": "Who was the first person to walk on the moon?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Historically documented space exploration milestone"
        }
    },
    {
        "qid": "SF009",
        "category": "simple_factual",
        "question": "What is the capital of Japan?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Clear geographical fact"
        }
    },
    {
        "qid": "SF010",
        "category": "simple_factual",
        "question": "What is the longest river in the world?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Geographic fact with scientific consensus"
        }
    },
    {
        "qid": "SF011",
        "category": "simple_factual",
        "question": "Who invented the telephone?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Well-documented historical invention"
        }
    },
    {
        "qid": "SF012",
        "category": "simple_factual",
        "question": "What is the smallest prime number?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Mathematical fact with a clear definition"
        }
    }
]

# Placeholder for other categories
NEGATION_ADVERSARIAL_QUESTIONS = [
    {
        "qid": "NA001",
        "category": "negation_adversarial",
        "question": "Is the sky not blue during the day?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Can be directly verified through observable facts"
        }
    },
    {
        "qid": "NA002",
        "category": "negation_adversarial",
        "question": "Are circles not round?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Mathematical definition can be clearly stated"
        }
    },
    {
        "qid": "NA003",
        "category": "negation_adversarial",
        "question": "Is water not a liquid at room temperature?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Verifiable scientific fact that can be tested"
        }
    },
    {
        "qid": "NA004",
        "category": "negation_adversarial",
        "question": "Can a triangle not have three sides?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Geometric definition is clear and unambiguous"
        }
    },
    {
        "qid": "NA005",
        "category": "negation_adversarial",
        "question": "Is gravity not a force of attraction?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Fundamental physics concept with clear definition"
        }
    },
    {
        "qid": "NA006",
        "category": "negation_adversarial",
        "question": "Are humans not mammals?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Biological classification is well-established"
        }
    },
    {
        "qid": "NA007",
        "category": "negation_adversarial",
        "question": "Is carbon not an element?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Clearly defined in the periodic table of elements"
        }
    },
    {
        "qid": "NA008",
        "category": "negation_adversarial",
        "question": "Can oxygen not be essential for human respiration?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Fundamental biological fact about human physiology"
        }
    },
    {
        "qid": "NA009",
        "category": "negation_adversarial",
        "question": "Is the Earth not spherical?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Scientifically proven planetary shape"
        }
    },
    {
        "qid": "NA010",
        "category": "negation_adversarial",
        "question": "Are photons not particles of light?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Well-established quantum physics concept"
        }
    },
    {
        "qid": "NA011",
        "category": "negation_adversarial",
        "question": "Is salt not a compound?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Chemical composition is well-defined"
        }
    },
    {
        "qid": "NA012",
        "category": "negation_adversarial",
        "question": "Can DNA not carry genetic information?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Fundamental biological principle of genetic inheritance"
        }
    }
]
IMPOSSIBLE_NONSENSE_QUESTIONS = [
    {
        "qid": "IN001",
        "category": "impossible_nonsense",
        "question": "What is the weight of a dream?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Dreams are abstract concepts without physical weight"
        }
    },
    {
        "qid": "IN002",
        "category": "impossible_nonsense",
        "question": "How do you measure the temperature of happiness?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Emotions are not physical properties that can be measured like temperature"
        }
    },
    {
        "qid": "IN003",
        "category": "impossible_nonsense",
        "question": "What color is silence?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Silence is an absence of sound, not a color"
        }
    },
    {
        "qid": "IN004",
        "category": "impossible_nonsense",
        "question": "How many unicorns can dance on the head of a pin?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Unicorns are mythical, and the question mixes impossible concepts"
        }
    },
    {
        "qid": "IN005",
        "category": "impossible_nonsense",
        "question": "What is the taste of pure mathematics?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Mathematics is an abstract concept without sensory properties"
        }
    },
    {
        "qid": "IN006",
        "category": "impossible_nonsense",
        "question": "How do you alphabetize a sunset?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Sunsets are natural phenomena that cannot be alphabetized"
        }
    },
    {
        "qid": "IN007",
        "category": "impossible_nonsense",
        "question": "What is the diameter of an idea?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Ideas are abstract concepts without physical dimensions"
        }
    },
    {
        "qid": "IN008",
        "category": "impossible_nonsense",
        "question": "How many emotions can fit into a teaspoon?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Emotions are not physical entities that can be measured by volume"
        }
    },
    {
        "qid": "IN009",
        "category": "impossible_nonsense",
        "question": "What is the gravitational pull of imagination?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Imagination is a mental process, not a physical force"
        }
    },
    {
        "qid": "IN010",
        "category": "impossible_nonsense",
        "question": "How do you knit a melody?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Melodies are auditory experiences, not physical objects that can be knitted"
        }
    },
    {
        "qid": "IN011",
        "category": "impossible_nonsense",
        "question": "What is the pH level of a metaphor?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Metaphors are linguistic constructs, not chemical solutions"
        }
    },
    {
        "qid": "IN012",
        "category": "impossible_nonsense",
        "question": "How many memories can fit inside a thunderstorm?",
        "gold_label": {
            "is_hallucination": True,
            "explanation": "Memories are abstract mental constructs, not physical entities that can be contained"
        }
    }
]
# Add other category lists here
COMMONSENSE_QUESTIONS = [
    {
        "qid": "CS001",
        "category": "commonsense",
        "question": "Would a fish survive on land?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Basic biological knowledge that fish require water to breathe"
        }
    },
    {
        "qid": "CS002",
        "category": "commonsense",
        "question": "Can a person breathe underwater without equipment?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Humans cannot breathe underwater without specialized equipment"
        }
    },
    {
        "qid": "CS003",
        "category": "commonsense",
        "question": "Is it possible to drink a hot beverage from an upside-down cup?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Gravity prevents liquid from staying in an inverted cup"
        }
    },
    {
        "qid": "CS004",
        "category": "commonsense",
        "question": "Would ice cream melt if left in direct sunlight?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Heat causes frozen substances to change state"
        }
    },
    {
        "qid": "CS005",
        "category": "commonsense",
        "question": "Can a rock float on water without any assistance?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Most rocks are denser than water and will sink"
        }
    },
    {
        "qid": "CS006",
        "category": "commonsense",
        "question": "Would a plant survive in complete darkness?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Plants require light for photosynthesis to survive"
        }
    },
    {
        "qid": "CS007",
        "category": "commonsense",
        "question": "Can a human walk on the ceiling without any special equipment?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Gravity prevents humans from walking upside down without specialized gear"
        }
    },
    {
        "qid": "CS008",
        "category": "commonsense",
        "question": "Would salt dissolve in oil?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Salt is water-soluble, not oil-soluble"
        }
    },
    {
        "qid": "CS009",
        "category": "commonsense",
        "question": "Can a bicycle move without someone pedaling it?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Bicycles require external force or momentum to move"
        }
    },
    {
        "qid": "CS010",
        "category": "commonsense",
        "question": "Would a paper umbrella protect someone from rain?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Paper disintegrates when wet and cannot provide effective protection"
        }
    },
    {
        "qid": "CS011",
        "category": "commonsense",
        "question": "Can a person survive without drinking water for a week?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Humans require water to survive; extended periods without water are fatal"
        }
    },
    {
        "qid": "CS012",
        "category": "commonsense",
        "question": "Would a candle burn underwater?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Fire requires oxygen and cannot burn underwater"
        }
    }
]

OBSCURE_LONG_TAIL_FACTUAL_QUESTIONS = [
    {
        "qid": "OLF001",
        "category": "obscure_long_tail_factual",
        "question": "What is the smallest known mammal in the world?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Kitti's hog-nosed bat (bumblebee bat) is scientifically recognized as the smallest mammal"
        }
    },
    {
        "qid": "OLF002",
        "category": "obscure_long_tail_factual",
        "question": "What is the rarest element on the periodic table?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Astatine is considered the rarest naturally occurring element"
        }
    },
    {
        "qid": "OLF003",
        "category": "obscure_long_tail_factual",
        "question": "What is the deepest known point in the ocean?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Challenger Deep in the Mariana Trench is the deepest known ocean point"
        }
    },
    {
        "qid": "OLF004",
        "category": "obscure_long_tail_factual",
        "question": "What is the longest word in the English language?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Pneumonoultramicroscopicsilicovolcanoconiosis is the longest word in major dictionaries"
        }
    },
    {
        "qid": "OLF005",
        "category": "obscure_long_tail_factual",
        "question": "What is the most isolated tree in the world?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "The Tree of Ténéré in Niger was the most isolated tree before it was destroyed"
        }
    },
    {
        "qid": "OLF006",
        "category": "obscure_long_tail_factual",
        "question": "What is the only continent without reptiles or snakes?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Antarctica is the only continent without native reptiles"
        }
    }
]

TIME_SENSITIVE_QUESTIONS = [
    {
        "qid": "TS001",
        "category": "time_sensitive",
        "question": "When did the World Health Organization declare COVID-19 a global pandemic?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "March 11, 2020"
        }
    },
    {
        "qid": "TS002",
        "category": "time_sensitive",
        "question": "Who won the 2020 United States presidential election?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Joe Biden"
        }
    },
    {
        "qid": "TS003",
        "category": "time_sensitive",
        "question": "In what year was the first mRNA COVID-19 vaccine authorized for public use?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "2020"
        }
    },
    {
        "qid": "TS004",
        "category": "time_sensitive",
        "question": "When did the James Webb Space Telescope begin full scientific operations?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "July 2022"
        }
    },
    {
        "qid": "TS005",
        "category": "time_sensitive",
        "question": "Which film won Best Picture at the 2021 Academy Awards?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Nomadland"
        }
    },
    {
        "qid": "TS006",
        "category": "time_sensitive",
        "question": "When was ChatGPT first released to the public as a research preview?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "November 2022"
        }
    },
    {
        "qid": "TS007",
        "category": "time_sensitive",
        "question": "Which city hosted the 2021 Summer Olympics after the one-year postponement?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Tokyo"
        }
    },
    {
        "qid": "TS008",
        "category": "time_sensitive",
        "question": "In what year did Apple release its first Macs with the M1 chip?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "2020"
        }
    },
    {
        "qid": "TS009",
        "category": "time_sensitive",
        "question": "When did NASA complete the DART asteroid-impact mission?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "September 2022"
        }
    },
    {
        "qid": "TS010",
        "category": "time_sensitive",
        "question": "In what year did Meta release the Quest 3 virtual reality headset?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "2023"
        }
    }
]

CONTEXT_DEPENDENT_QUESTIONS = [
    {
        "qid": "CD001",
        "category": "context_dependent",
        "context": "The Solaris Research Station was established in 2023 to study radiation patterns on the lunar surface. Its team is composed of scientists from Canada, Japan, and Brazil. The station sends daily reports to Earth using a new encrypted communication system developed specifically for the mission.",
        "question": "Which three countries have scientists working at the Solaris Research Station?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Answer can be directly found in the provided context"
        }
    },
    {
        "qid": "CD002",
        "category": "context_dependent",
        "context": "City officials approved the Aurora Metro Line expansion in 2024 after years of delays. The new section will add five stations to the north side of the city, with construction expected to finish in late 2026. Funding for the project came partly from a transportation bond passed by voters in 2022.",
        "question": "How many new stations are being added to the Aurora Metro Line?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Answer can be directly found in the provided context"
        }
    },
    {
        "qid": "CD003",
        "category": "context_dependent",
        "context": "The National Archive Restoration Initiative launched in 2021 to digitize fragile documents from the early 1800s. Volunteers and historians work together to scan letters, maps, and journals. As of 2024, the project has preserved more than 18,000 items, but its organizers expect the total to exceed 25,000 by the time the initiative concludes.",
        "question": "When did the National Archive Restoration Initiative launch?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Answer can be directly found in the provided context"
        }
    },
    {
        "qid": "CD004",
        "category": "context_dependent",
        "context": "Marine biologists conducted a three-year study of the Crescent Reef ecosystem, beginning in 2020. Their research showed a 12% increase in coral regrowth after a targeted restoration program was introduced. The team attributed the success to reduced pollution and a temporary ban on fishing in surrounding waters.",
        "question": "How long did the Crescent Reef study last?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Answer can be directly found in the provided context"
        }
    },
    {
        "qid": "CD005",
        "category": "context_dependent",
        "context": "In 2025, the Maplewood Public Library completed a major technology upgrade that replaced all public computers and introduced self-checkout kiosks. The project was funded by a state education grant and private donations. The library plans a second phase next year to redesign the children's reading area.",
        "question": "What new technology did the Maplewood Public Library introduce during its 2025 upgrade?",
        "gold_label": {
            "is_hallucination": False,
            "explanation": "Answer can be directly found in the provided context"
        }
    }
]

def load_question_set():
    # Person A moved the question set to the data/ directory
    path = Path("data/question_set_v1.json")

    if not path.exists():
        raise FileNotFoundError(
            f"Expected question set at {path}, but it does not exist. "
            "Make sure the file is committed by Person A."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data