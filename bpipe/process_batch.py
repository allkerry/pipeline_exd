import logging
from importlib import metadata
from datetime import datetime, timezone
import cupy as cp
import numpy as np
import gc
from exorde_data import (
    ProtocolItem,
    ProtocolAnalysis,
    ProcessedItem,
    Batch,
    Content,
    BatchKindEnum,
    CollectionClientVersion,
    CollectedAt,
    CollectionModule,
    Processed,
    Analysis,
    Classification,
    Keywords,
    LanguageScore,
    Sentiment,
    Embedding,
    SourceType,
    TextType,
    Emotion,
    Irony,
    Age,
    Gender,
    Analysis
)
from exorde_data import Url

try:
    from exorde_data import Username, UserProfileUrl
except (ImportError, AttributeError):
    class Username(str):
        pass

    class UserProfileUrl(str):
        pass

from tag import tag
from collections import Counter

def clear_cupy_memory():
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
        logging.debug("CuPy memory cleared")
    except Exception as e:
        logging.warning(f"Error clearing CuPy memory: {e}")

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def merge_chunks(chunks: list[ProcessedItem]) -> ProcessedItem:
    try:
        if len(chunks) == 1:
            return chunks[0]

        categories_list = []
        top_keywords_list = []
        gender_list = []
        sentiment_list = []
        source_type_list = []
        text_type_list = []
        emotion_list = []
        language_score_list = []
        irony_list = []
        age_list = []
        embedding_list = []

        logging.info(f"[Item merging] Merging {len(chunks)} chunks.")

        try:
            for processed_item in chunks:
                item_analysis_ = processed_item.analysis
                categories_list.append(item_analysis_.classification)
                top_keywords_list.append(item_analysis_.top_keywords)
                gender_list.append(item_analysis_.gender)
                sentiment_list.append(item_analysis_.sentiment)
                source_type_list.append(item_analysis_.source_type)
                text_type_list.append(item_analysis_.text_type)
                emotion_list.append(item_analysis_.emotion)
                language_score_list.append(item_analysis_.language_score)
                irony_list.append(item_analysis_.irony)
                age_list.append(item_analysis_.age)

                try:
                    embedding_array = cp.array(item_analysis_.embedding.vector)
                    embedding_list.append(embedding_array)
                except cp.cuda.memory.OutOfMemoryError:
                    logging.warning("GPU memory low, using numpy for embeddings")
                    embedding_array = np.array(item_analysis_.embedding.vector)
                    embedding_list.append(embedding_array)

            most_common_category = Most_Common([x.label for x in categories_list])
            category_aggregated = Classification(
                label=most_common_category,
                score=max([x.score for x in categories_list]),
            )

            top_keywords_aggregated = list(set([kw for keywords in top_keywords_list for kw in keywords.keywords]))
            top_keywords_aggregated = Keywords(top_keywords_aggregated)

            try:
                gender_aggregated = Gender(
                    male=float(cp.median(cp.array([x.male for x in gender_list]))),
                    female=float(cp.median(cp.array([x.female for x in gender_list]))),
                )
                sentiment_aggregated = Sentiment(float(cp.median(cp.array(sentiment_list))))

                clear_cupy_memory()

            except (cp.cuda.memory.OutOfMemoryError, RuntimeError):
                logging.warning("GPU memory insufficient, using numpy for aggregation")
                gender_aggregated = Gender(
                    male=float(np.median([x.male for x in gender_list])),
                    female=float(np.median([x.female for x in gender_list])),
                )
                sentiment_aggregated = Sentiment(float(np.median(sentiment_list)))

            source_type_aggregated = SourceType(Most_Common(source_type_list))

            try:
                text_type_aggregated = TextType(
                    assumption=float(cp.median(cp.array([tt.assumption for tt in text_type_list]))),
                    anecdote=float(cp.median(cp.array([tt.anecdote for tt in text_type_list]))),
                    none=float(cp.median(cp.array([tt.none for tt in text_type_list]))),
                    definition=float(cp.median(cp.array([tt.definition for tt in text_type_list]))),
                    testimony=float(cp.median(cp.array([tt.testimony for tt in text_type_list]))),
                    other=float(cp.median(cp.array([tt.other for tt in text_type_list]))),
                    study=float(cp.median(cp.array([tt.study for tt in text_type_list]))),
                )
                clear_cupy_memory()

            except (cp.cuda.memory.OutOfMemoryError, RuntimeError):
                text_type_aggregated = TextType(
                    assumption=float(np.median([tt.assumption for tt in text_type_list])),
                    anecdote=float(np.median([tt.anecdote for tt in text_type_list])),
                    none=float(np.median([tt.none for tt in text_type_list])),
                    definition=float(np.median([tt.definition for tt in text_type_list])),
                    testimony=float(np.median([tt.testimony for tt in text_type_list])),
                    other=float(np.median([tt.other for tt in text_type_list])),
                    study=float(np.median([tt.study for tt in text_type_list])),
                )

            try:
                emotion_aggregated = Emotion(
                    love=float(cp.median(cp.array([e.love for e in emotion_list]))),
                    admiration=float(cp.median(cp.array([e.admiration for e in emotion_list]))),
                    joy=float(cp.median(cp.array([e.joy for e in emotion_list]))),
                    approval=float(cp.median(cp.array([e.approval for e in emotion_list]))),
                    caring=float(cp.median(cp.array([e.caring for e in emotion_list]))),
                    excitement=float(cp.median(cp.array([e.excitement for e in emotion_list]))),
                    gratitude=float(cp.median(cp.array([e.gratitude for e in emotion_list]))),
                    desire=float(cp.median(cp.array([e.desire for e in emotion_list]))),
                    anger=float(cp.median(cp.array([e.anger for e in emotion_list]))),
                    optimism=float(cp.median(cp.array([e.optimism for e in emotion_list]))),
                    disapproval=float(cp.median(cp.array([e.disapproval for e in emotion_list]))),
                    grief=float(cp.median(cp.array([e.grief for e in emotion_list]))),
                    annoyance=float(cp.median(cp.array([e.annoyance for e in emotion_list]))),
                    pride=float(cp.median(cp.array([e.pride for e in emotion_list]))),
                    curiosity=float(cp.median(cp.array([e.curiosity for e in emotion_list]))),
                    neutral=float(cp.median(cp.array([e.neutral for e in emotion_list]))),
                    disgust=float(cp.median(cp.array([e.disgust for e in emotion_list]))),
                    disappointment=float(cp.median(cp.array([e.disappointment for e in emotion_list]))),
                    realization=float(cp.median(cp.array([e.realization for e in emotion_list]))),
                    fear=float(cp.median(cp.array([e.fear for e in emotion_list]))),
                    relief=float(cp.median(cp.array([e.relief for e in emotion_list]))),
                    confusion=float(cp.median(cp.array([e.confusion for e in emotion_list]))),
                    remorse=float(cp.median(cp.array([e.remorse for e in emotion_list]))),
                    embarrassment=float(cp.median(cp.array([e.embarrassment for e in emotion_list]))),
                    surprise=float(cp.median(cp.array([e.surprise for e in emotion_list]))),
                    sadness=float(cp.median(cp.array([e.sadness for e in emotion_list]))),
                    nervousness=float(cp.median(cp.array([e.nervousness for e in emotion_list]))),
                )
                clear_cupy_memory()

            except (cp.cuda.memory.OutOfMemoryError, RuntimeError):
                emotion_aggregated = Emotion(
                    love=float(np.median([e.love for e in emotion_list])),
                    admiration=float(np.median([e.admiration for e in emotion_list])),
                    joy=float(np.median([e.joy for e in emotion_list])),
                    approval=float(np.median([e.approval for e in emotion_list])),
                    caring=float(np.median([e.caring for e in emotion_list])),
                    excitement=float(np.median([e.excitement for e in emotion_list])),
                    gratitude=float(np.median([e.gratitude for e in emotion_list])),
                    desire=float(np.median([e.desire for e in emotion_list])),
                    anger=float(np.median([e.anger for e in emotion_list])),
                    optimism=float(np.median([e.optimism for e in emotion_list])),
                    disapproval=float(np.median([e.disapproval for e in emotion_list])),
                    grief=float(np.median([e.grief for e in emotion_list])),
                    annoyance=float(np.median([e.annoyance for e in emotion_list])),
                    pride=float(np.median([e.pride for e in emotion_list])),
                    curiosity=float(np.median([e.curiosity for e in emotion_list])),
                    neutral=float(np.median([e.neutral for e in emotion_list])),
                    disgust=float(np.median([e.disgust for e in emotion_list])),
                    disappointment=float(np.median([e.disappointment for e in emotion_list])),
                    realization=float(np.median([e.realization for e in emotion_list])),
                    fear=float(np.median([e.fear for e in emotion_list])),
                    relief=float(np.median([e.relief for e in emotion_list])),
                    confusion=float(np.median([e.confusion for e in emotion_list])),
                    remorse=float(np.median([e.remorse for e in emotion_list])),
                    embarrassment=float(np.median([e.embarrassment for e in emotion_list])),
                    surprise=float(np.median([e.surprise for e in emotion_list])),
                    sadness=float(np.median([e.sadness for e in emotion_list])),
                    nervousness=float(np.median([e.nervousness for e in emotion_list])),
                )

            try:
                language_score_aggregated = LanguageScore(float(cp.median(cp.array(language_score_list))))

                irony_aggregated = Irony(
                    irony=float(cp.median(cp.array([i.irony for i in irony_list]))),
                    non_irony=float(cp.median(cp.array([i.non_irony for i in irony_list]))),
                )

                _age_agg = Age(
                    below_twenty=float(cp.median(cp.array([a.below_twenty for a in age_list]))),
                    twenty_thirty=float(cp.median(cp.array([a.twenty_thirty for a in age_list]))),
                    thirty_forty=float(cp.median(cp.array([a.thirty_forty for a in age_list]))),
                    forty_more=float(cp.median(cp.array([a.forty_more for a in age_list]))),
                )
                age_aggregated = _age_agg if sum([_age_agg.below_twenty, _age_agg.twenty_thirty, _age_agg.thirty_forty, _age_agg.forty_more]) > 0 else Age(below_twenty=0.25, twenty_thirty=0.25, thirty_forty=0.25, forty_more=0.25)
                clear_cupy_memory()

            except (cp.cuda.memory.OutOfMemoryError, RuntimeError):
                language_score_aggregated = LanguageScore(float(np.median(language_score_list)))

                irony_aggregated = Irony(
                    irony=float(np.median([i.irony for i in irony_list])),
                    non_irony=float(np.median([i.non_irony for i in irony_list])),
                )

                _age_agg = Age(
                    below_twenty=float(np.median([a.below_twenty for a in age_list])),
                    twenty_thirty=float(np.median([a.twenty_thirty for a in age_list])),
                    thirty_forty=float(np.median([a.thirty_forty for a in age_list])),
                    forty_more=float(np.median([a.forty_more for a in age_list])),
                )
                age_aggregated = _age_agg if sum([_age_agg.below_twenty, _age_agg.twenty_thirty, _age_agg.thirty_forty, _age_agg.forty_more]) > 0 else Age(below_twenty=0.25, twenty_thirty=0.25, thirty_forty=0.25, forty_more=0.25)

            try:
                if embedding_list:
                    if isinstance(embedding_list[0], cp.ndarray):
                        embedding_stack = cp.stack(embedding_list)
                        centroid_vector = cp.median(embedding_stack, axis=0)
                        distances = [cp.linalg.norm(emb - centroid_vector) for emb in embedding_list]
                        closest_idx = cp.argmin(cp.array(distances))
                        closest_embedding = Embedding(list(embedding_list[int(closest_idx)].get().astype(float)))
                    else:
                        embedding_stack = np.stack(embedding_list)
                        centroid_vector = np.median(embedding_stack, axis=0)
                        distances = [np.linalg.norm(emb - centroid_vector) for emb in embedding_list]
                        closest_idx = np.argmin(distances)
                        closest_embedding = Embedding(list(embedding_list[closest_idx].astype(float)))
                else:
                    closest_embedding = Embedding([0.0] * 384)

                clear_cupy_memory()

            except (cp.cuda.memory.OutOfMemoryError, RuntimeError, Exception):
                logging.warning("Error in embedding aggregation, using default")
                closest_embedding = Embedding([0.0] * 384)

            _merged_analysis = ProtocolAnalysis(
                classification=category_aggregated,
                top_keywords=top_keywords_aggregated,
                language_score=language_score_aggregated,
                sentiment=sentiment_aggregated,
                embedding=closest_embedding,
                source_type=source_type_aggregated,
                emotion=emotion_aggregated,
            )
            _merged_analysis["gender"] = {"male": gender_aggregated.male, "female": gender_aggregated.female}
            _merged_analysis["text_type"] = {
                "assumption": text_type_aggregated.assumption,
                "anecdote": text_type_aggregated.anecdote,
                "none": text_type_aggregated.none,
                "definition": text_type_aggregated.definition,
                "testimony": text_type_aggregated.testimony,
                "other": text_type_aggregated.other,
                "study": text_type_aggregated.study,
            }
            _merged_analysis["irony"] = {"irony": irony_aggregated.irony, "non_irony": irony_aggregated.non_irony}
            _merged_analysis["age"] = {
                "below_twenty": age_aggregated.below_twenty,
                "twenty_thirty": age_aggregated.twenty_thirty,
                "thirty_forty": age_aggregated.thirty_forty,
                "forty_more": age_aggregated.forty_more,
            }

            merged_item = ProcessedItem(
                item=chunks[0].item,
                analysis=_merged_analysis,
                collection_client_version=chunks[0].collection_client_version,
                collection_module=chunks[0].collection_module,
                collected_at=chunks[0].collected_at,
            )

        except Exception as e:
            logging.exception(f"[Merging items chunks] ERROR:\n {e}")
            merged_item = None

        finally:
            clear_cupy_memory()

    except Exception as e:
        logging.exception(f"[Merging items chunks] ERROR:\n {e}")
        merged_item = None
    return merged_item

SOCIAL_DOMAINS = [
    "4chan.org",
    "4channel.org",
    "reddit.com",
    "twitter.com",
    "bsky.app",
    "t.com",
    "x.com",
    "youtube.com",
    "yt.co",
    "lemmy.world",
    "mastodon.social",
    "weibo.com",
    "nostr.social",
    "nostr.com",
    "jeuxvideo.com",
    "forocoches.com",
    "bitcointalk.org",
    "ycombinator.com",
    "news.ycombinator.com",
    "tradingview.com",
    "followin.in",
    "seekingalpha.io",
    "threads.net",
    "telegram.org",
    "tumblr.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "whatsapp.com",
    "9gag.com",
    "techhaven.org",
    "hey.xyz",
    "dscvr.one",
    "warpcast.com",
    "discord.com",
    "matrix.org",
    "gab.com",
    "parler.com",
    "truthsocial.com",
    "vk.com",
    "vero.co",
    "substack.com",
    "8kun.top",
    "ello.co",
    "minds.com",
    "mewe.com",
    "livejournal.com",
    "plurk.com",
    "friendica.net",
    "aminoapps.com",
    "bitchute.com",
    "gettr.com",
    "odysee.com",
    "poal.co",
    "rumble.com",
    "wimkin.com",
    "twitch.tv",
    "rocket.chat",
    "snap.com",
    "vidlii.com"
]


def get_source_type(item: ProtocolItem) -> SourceType:
    domain = (item.domain or "").lower()
    # Точное совпадение "twitter.com" пропускало "www.twitter.com" / "m.reddit.com" —
    # теперь match по суффиксу домена.
    if any(domain == d or domain.endswith("." + d) for d in SOCIAL_DOMAINS):
        return SourceType("social")
    return SourceType("news")

from opentelemetry import trace
from opentelemetry.trace import StatusCode

def process_batch(
    batch: list[tuple[int, Processed]], lab_configuration
) -> Batch:
    tracer = trace.get_tracer(__name__)
    logging.info(f"[BATCH] Processing {len(batch)} items")
    with tracer.start_as_current_span("tag") as tag_span:
        analysis_results: list[Analysis] = tag(
            [processed.translation.translation for (__id__, processed) in batch],
            lab_configuration,
        )
        tag_span.set_status(StatusCode.OK)

    complete_processes: dict[int, list[ProcessedItem]] = {}
    for (id, processed), analysis in zip(batch, analysis_results):
        prot_item: ProtocolItem = ProtocolItem(
            raw_content=Content(processed.item.content),
            translated_content=Content(processed.translation.translation),
            created_at=processed.item.created_at,
            domain=processed.item.domain,
            url=Url(processed.item.url),
            language=processed.translation.language,
        )

        if processed.item.title:
            prot_item['title'] = processed.item.title
        if processed.item.summary:
            prot_item['summary'] = processed.item.summary
        if processed.item.picture:
            prot_item['picture'] = processed.item.picture
        if processed.item.author:
            prot_item['author'] = processed.item.author
        if processed.item.external_id:
            prot_item['external_id'] = processed.item.external_id
        if processed.item.external_parent_id:
            prot_item['external_parent_id'] = processed.item.external_parent_id

        if 'username' in processed.item and processed.item['username']:
            prot_item['username'] = Username(processed.item['username'])

        if 'userprofile_url' in processed.item and processed.item['userprofile_url']:
            prot_item['userprofile_url'] = UserProfileUrl(processed.item['userprofile_url'])

        # ProtocolAnalysis в установленной версии exorde_data не содержит
        # gender/text_type/irony/age — они инжектируются напрямую в dict после создания
        _analysis = ProtocolAnalysis(
            classification=analysis.classification,
            top_keywords=processed.top_keywords,
            language_score=analysis.language_score,
            sentiment=analysis.sentiment,
            embedding=analysis.embedding,
            source_type=get_source_type(prot_item),
            emotion=analysis.emotion,
        )
        _analysis["gender"] = {"male": analysis.gender.male, "female": analysis.gender.female}
        _analysis["text_type"] = {
            "assumption": analysis.text_type.assumption,
            "anecdote": analysis.text_type.anecdote,
            "none": analysis.text_type.none,
            "definition": analysis.text_type.definition,
            "testimony": analysis.text_type.testimony,
            "other": analysis.text_type.other,
            "study": analysis.text_type.study,
        }
        _analysis["irony"] = {"irony": analysis.irony.irony, "non_irony": analysis.irony.non_irony}
        _analysis["age"] = {
            "below_twenty": analysis.age.below_twenty,
            "twenty_thirty": analysis.age.twenty_thirty,
            "thirty_forty": analysis.age.thirty_forty,
            "forty_more": analysis.age.forty_more,
        }

        completed: ProcessedItem = ProcessedItem(
            item=prot_item,
            analysis=_analysis,
            collection_client_version=CollectionClientVersion(
                f"exorde:v.{metadata.version('exorde_data')}"
            ),
            collection_module=CollectionModule("micro"),
            collected_at=CollectedAt(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"),
        )
        if not complete_processes.get(id, {}):
            complete_processes[id] = []
        complete_processes[id].append(completed)
    aggregated = []
    with tracer.start_as_current_span("merge_chunks") as merge_chunks_span:
        for __key__, values in complete_processes.items():
            merged_ = merge_chunks(values)
            if merged_ is not None:
                aggregated.append(merged_)
        merge_chunks_span.set_status(StatusCode.OK)
    result_batch: Batch = Batch(
        items=aggregated,
        kind=BatchKindEnum.SPOTTING
    )
    return result_batch
