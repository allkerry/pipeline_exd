from exorde_data import Translation, Language, Translated, Item

def translate(item: Item, installed_languages, low_memory: bool = False) -> Translation:
    return Translation(
        language=Language("en"),
        translation=Translated(str(item.content)),
    )
