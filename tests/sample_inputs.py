"""Hard-coded English SNS samples. Swap to a real dataset later by changing the loader,
not the downstream shape: each sample is {id, text}.

The extended set covers: different speech acts, registers, emotions, and cultural refs
that should exercise dict, retriever fuzzy, and web+llm paths."""

SAMPLES: list[dict] = [
    # Core 5 (dict-heavy)
    {"id": "s1", "text": "omg just got back from Trader Joe's, pumpkin spice EVERYTHING lmaooo 🎃"},
    {"id": "s2", "text": "venmo me $20 for the uber home from the frat party"},
    {"id": "s3", "text": "crying at the SNL cold open rn"},
    {"id": "s4", "text": "prom tmrw and I have NOTHING to wear 😭😭😭"},
    {"id": "s5", "text": "happy thanksgiving fam, eating way too much turkey"},
    # Retriever-targeting: fuzzy variants of dict entries
    {"id": "s6", "text": "Who's watching the Super Bowl halftime show?? Taylor is performing rumors"},
    {"id": "s7", "text": "Christmas Eve dinner with the fam, too much food lol"},
    {"id": "s8", "text": "ordered a grande pumpkin spice latte at starbucks, feeling basic"},
    # OOD / web+llm targeting: should force fresh retrieval
    {"id": "s9", "text": "the sigma ohio rizz on this one is lowkey insane"},
    {"id": "s10", "text": "on my tinder after another situationship collapse haha"},
    # Register/speech_act diversity
    {"id": "s11", "text": "Dear hiring manager, I am writing to express my interest in the position."},
    {"id": "s12", "text": "PLEASE tell me you also think this homework is ridiculous???"},
    {"id": "s13", "text": "Can you send me the zoom link? meeting starts in 5"},
    {"id": "s14", "text": "Congrats on the new job!!! So proud of you 🥳🎉"},
    {"id": "s15", "text": "literally nobody: / me at 3am: should i text my ex /s"},
    # Emotion extremes
    {"id": "s16", "text": "I CANNOT believe they cancelled the show 😤😤 absolutely furious"},
    {"id": "s17", "text": "passed my SAT with a perfect score!!! grinding paid off"},
    {"id": "s18", "text": "scrolling tiktok at 2am wondering what i'm doing with my life"},
    # Culturally dense
    {"id": "s19", "text": "black friday haul at the mall was insane, got everything half off"},
    {"id": "s20", "text": "watching the superbowl with a pint of häagen-dazs, living the dream"},
]

OOD_TERM = "skibidi"
