import json

dataset = [
    {"slang": "Yo, c'est quoi ton souci avec ta box ?",
     "neutral": "Bonjour, avez-vous un problème avec votre box ?",
     "formal": "Bonjour, en quoi puis-je vous aider concernant votre connexion internet ?"},

    {"slang": "Attends, j’check ça vite fait.",
     "neutral": "Un instant, je vais vérifier cela.",
     "formal": "Veuillez patienter un instant, je vais procéder à la vérification."},

    {"slang": "Désolé mec, j’peux rien faire, c’est mort.",
     "neutral": "Je suis désolé, mais ce n’est pas possible.",
     "formal": "Je suis navré, mais cette demande ne peut être satisfaite en raison des règles en vigueur."},

    {"slang": "Oups, ma faute ! Je répare ça direct.",
     "neutral": "Je suis désolé pour l’erreur, je vais corriger cela immédiatement.",
     "formal": "Je vous prie de bien vouloir excuser cette erreur, je vais la rectifier immédiatement."},

    {"slang": "T’inquiète, je gère.",
     "neutral": "Ne vous inquiétez pas, je m’en occupe.",
     "formal": "Soyez assuré que je prends en charge cette demande immédiatement."},

    {"slang": "C'est le bazar, faut que je regarde ça.",
     "neutral": "Il semble y avoir un souci, je vais vérifier cela.",
     "formal": "Un problème semble être survenu, permettez-moi d'effectuer une vérification."},

    {"slang": "Reste en ligne, on bidouille un truc.",
     "neutral": "Veuillez rester en ligne, nous sommes en train de résoudre le problème.",
     "formal": "Je vous remercie de patienter pendant que nous procédons à la résolution du problème."},

    {"slang": "Désolé, mais là, c’est la galère. J’peux rien faire.",
     "neutral": "Je suis navré, mais je ne peux rien faire pour cela.",
     "formal": "Je suis au regret de vous informer que nous ne pouvons malheureusement pas donner suite à cette requête."},

    {"slang": "Ça part en vrille, je vais régler ça.",
     "neutral": "Il semble y avoir un souci, je vais le corriger.",
     "formal": "Il semble y avoir une anomalie, permettez-moi d’intervenir pour la résoudre."},

    {"slang": "Dépêche-toi, j’ai pas toute la journée !",
     "neutral": "Pouvez-vous me dire ce qui ne va pas ?",
     "formal": "Puis-je vous demander quel est le problème que vous rencontrez aujourd’hui ?"}
]

with open("call_center_slang_fr.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print("Dataset enregistré sous 'call_center_slang_fr.json'")
