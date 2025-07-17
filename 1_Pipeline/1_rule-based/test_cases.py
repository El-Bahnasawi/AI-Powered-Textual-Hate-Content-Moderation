test_cases_1 = [
    "That party was lit AF 🔥",                        # Casual use of 🔥
    "I'm dead 💀💀💀🤣",                                # 💀 as a meme (not hate)
    "He went nuclear on that code review! 💣",         # 💣 used metaphorically
    "She roasted him in the debate.",                 # 'roast' ≠ insult
    "Bruh, you’re such a clown 🤡",                    # 🤡 = meme, not hate
    "Kill it in your next exam!",                     # Positive use of 'kill'
    "Build the wall of inspiration, brick by brick.", # Metaphorical use of a politicized phrase
    "Go back to basics, that’s where the magic is.",  # Benign 'go back' phrase
    "KKK... Keyboard, Keycap, and Keystroke",         # Humorous acronym collision
    "This food is so good, it’s criminal!",            # Hyperbole, not threat
    "You're a beast at coding!",                       # Compliment, not dehumanization
    "Protect your peace and energy.",                  # Not white nationalist dogwhistle
    "We need a clean sweep this quarter 🔫",           # Gun emoji used metaphorically
    "I bombed that test 💣",                           # Self-deprecating, harmless
    "Damn, she's killing it on stage!",                # Slang for doing great
    ]

test_cases_2 = [
    # ✅ SAFE — should be passed by both rule-based and ML model
    "What a beautiful day to go for a walk!",
    "Congratulations on your promotion, you earned it!",
    "I love seeing diverse voices represented in tech.",
    "Thanks for the help today — you're awesome!",
    "Enjoying a quiet evening with my cat 🐱 and tea ☕",
    "Learning new languages is so fulfilling.",
    
    # 🔥 RULE-BASED — should be caught by rule-based filter
    "Go back to your country.",
    "They're nothing but parasites and subhumans.",
    "Kill them all, I don't care anymore.",
    "White power is the only way forward.",
    "You're such a slut and a whore.",
    "Build the wall now! Deport all of them!",
    "KKK was right about them.",
    "F***ing Muslims ruin everything. 💣",
    "These invaders should be wiped out.",
    "Roasties like her should stay silent.",
    
    # 🧠 ML-DETECTABLE — implicit/coded hate, needs semantic model to flag
    "I’m not racist, but some people just don't belong here.",
    "They always play the victim and expect handouts.",
    "She only got the job because of diversity quotas.",
    "Ever notice how certain neighborhoods feel unsafe?",
    "They may look human, but they’re not like us.",
    "You can’t fix their kind — it’s in their culture.",
    "We need to protect our bloodlines and traditions.",
    "Funny how crime goes up when they move in.",
    "Some cultures are fundamentally incompatible with democracy.",
    "They act civilized, but you know the truth."
    ]