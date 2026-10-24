from bioclip import Rank

PLAUSIBLE_TAXA = {
    # FOREGROUND: Wintergreen Lure Attractants (Beneficials & Specific Pests)
    Rank.FAMILY: [
        "Chrysopidae",      # Green Lacewings (Beneficial - voracious mealybug/aphid predators)
        "Syrphidae",        # Hoverflies / Syrphid Flies (Beneficial predators)
        "Pentatomidae",     # Stink Bugs (e.g., Brown Marmorated Stink Bug - PEST)
        "Vespidae",         # Yellowjackets and Paper Wasps (Attracted to chemical lures)
        "Miridae",          # Plant bugs / Leaf-hopper relatives
        "Cicadellidae",     # Leafhoppers (Pierce's Disease vectors in background/foreground)
        "Pseudococcidae"    # Mealybugs (Vine mealybug)
    ],
    
    # BACKGROUND: Macro Vineyard Threats (Triggers via UniFi Native Animal Webhook)
    Rank.ORDER: [
        "Artiodactyla",     # Black-tailed Deer (Canopy browsing)
        "Rodentia"          # Gophers/Squirrels (Root/vine damage)
    ],
    Rank.CLASS: [
        "Aves"              # Starlings / Finches (Fruit-destroying flocks)
    ]
}
