from exorde_data import (
    Analysis, Age, Author, Batch, BatchKindEnum, Classification,
    CollectedAt, CollectionClientVersion, CollectionModule,
    Content, CreatedAt, Domain, Embedding, Emotion, ExternalId,
    ExternalParentId, Gender, Irony, Item, Keywords, Language,
    LanguageScore, Processed, ProcessedItem, ProtocolAnalysis,
    ProtocolItem, Sentiment, SourceType, TextType, Title,
    Translated, Translation, Url,
)

# Этих типов нет в full-ветке, делаем заглушки
class Username(str): pass
class UserProfileUrl(str): pass
