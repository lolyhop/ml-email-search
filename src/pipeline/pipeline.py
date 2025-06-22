import time
import uuid
import logging
import typing as tp

from src.retrievers.base import Embedder, EmbedderConfig
from src.retrievers.dense import DenseEmbedder
from src.retrievers.dummy import DummyEmbedder
from src.index.base import BaseIndex
from src.index.factory import IndexFactory
from src.index.config import IndexConfig
from src.index.entity import FaissEntity, FaissCorpus
from src.pipeline.config import PipelineConfig, SearchResult, PipelineResult

logger = logging.getLogger(__name__)


class SearchPipeline:
    """Pipeline for running end-to-end search experiment."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config: PipelineConfig = config
        self.embedder: tp.Optional[Embedder] = None
        self.corpus: tp.Optional[FaissCorpus] = None
        self.index: tp.Optional[BaseIndex] = None
        self.build_time: float = 0.0

    def _create_embedder(self) -> Embedder:
        """Create and initialize embedder."""
        if self.config.embedder_config.model_name == "dummy_model":
            embedder = DummyEmbedder(self.config.embedder_config)
        else:
            embedder = DenseEmbedder(self.config.embedder_config)

        embedder.initialize()
        return embedder

    def _create_corpus(self) -> FaissCorpus:
        """Create corpus from documents."""
        entities = []
        for doc_id, content in self.config.documents:
            if isinstance(doc_id, str):
                doc_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
            else:
                doc_uuid = doc_id

            entities.append(FaissEntity(id=doc_uuid, content=content))

        return FaissCorpus(entities=entities)

    def setup(self) -> None:
        """Setup embedder and corpus."""
        logger.info("Setting up pipeline...")

        logger.info("Initializing embedder...")
        self.embedder = self._create_embedder()
        logger.info("Embedder initialized!")

        logger.info("Initializing corpus...")
        self.corpus = self._create_corpus()
        logger.info(f"Corpus initialized with {self.corpus.n_documents} documents!")

    def build_index(self) -> None:
        """Build the configured index."""
        if self.embedder is None or self.corpus is None:
            raise RuntimeError("Pipeline not setup. Call setup() first.")

        logger.info(f"Building {self.config.index_type} index...")

        start_time = time.time()

        # Create and build index
        self.index = IndexFactory.create(
            self.config.index_type, self.config.index_config
        )
        self.index.build_from_corpus(self.corpus, self.embedder)

        self.build_time = time.time() - start_time
        logger.info(f"Index built in {self.build_time:.3f}s")

    def run_queries(self) -> PipelineResult:
        """Run all queries on the built index."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        logger.info("Running queries...")

        # Embed queries
        query_embeddings = self.embedder.embed_query(self.config.queries)
        if hasattr(query_embeddings, "cpu"):
            query_embeddings = query_embeddings.cpu().numpy()

        # Run searches
        results = []
        total_search_time = 0.0

        for i, query in enumerate(self.config.queries):
            query_embedding = query_embeddings[i : i + 1]  # Single query

            # Time the search
            start_time = time.time() if self.config.measure_time else None

            scores, doc_ids = self.index.search_with_ids(query_embedding, self.config.k)

            search_time = time.time() - start_time if start_time else None
            if search_time:
                total_search_time += search_time

            # Get full documents
            indices = [i for i in range(min(len(scores[0]), len(doc_ids[0])))]
            documents = self.index.get_entities_by_indices(indices)

            results.append(
                SearchResult(
                    query=query,
                    scores=scores[0],
                    document_ids=doc_ids[0],
                    documents=documents,
                    search_time=search_time,
                )
            )

        avg_search_time = (
            total_search_time / len(self.config.queries) if self.config.queries else 0.0
        )

        logger.info(
            f"Completed {len(self.config.queries)} queries in {total_search_time:.3f}s"
        )

        return PipelineResult(
            index_type=self.config.index_type,
            build_time=self.build_time,
            total_search_time=total_search_time,
            avg_search_time=avg_search_time,
            results=results,
        )

    def run_all(self) -> PipelineResult:
        """Run the complete pipeline."""
        self.setup()
        self.build_index()
        return self.run_queries()

    def print_results(self, result: PipelineResult) -> None:
        """Print formatted results."""
        print("\n" + "=" * 60)
        print("PIPELINE RESULTS")
        print("=" * 60)

        print(f"Index Type: {result.index_type}")
        print(f"Build Time: {result.build_time:.3f}s")
        print(f"Total Search Time: {result.total_search_time:.3f}s")
        print(f"Average Search Time: {result.avg_search_time:.6f}s")
        print(f"Queries Processed: {len(result.results)}")

        if result.results:
            print(f"\nSearch Results:")
            print("-" * 60)

            for query_result in result.results[:2]:
                print(f"\nQuery: '{query_result.query}'")
                print(
                    f"Search Time: {query_result.search_time:.6f}s"
                    if query_result.search_time
                    else ""
                )

                for i, doc in enumerate(query_result.documents[:3]):
                    score = query_result.scores[i]
                    print(f"  {i+1}. [{score:.4f}] {doc.content[:60]}...")


def create_sample_pipeline(index_type: str = "hnsw") -> SearchPipeline:
    """Create a sample pipeline for testing."""

    documents = [
        ("doc1", "Meeting tomorrow at 3pm with the engineering team"),
        ("doc2", "Project deadline is next week, need to finish coding"),
        ("doc3", "Lunch reservation confirmed for Friday at noon"),
        ("doc4", "Code review scheduled for this afternoon"),
        ("doc5", "Team building event planned for next month"),
    ]

    queries = [
        "team meeting",
        "project deadline",
        "lunch plans",
    ]

    # TODO: Add real maintainance of 2 heads embedders
    embedder_config = EmbedderConfig(
        model_name="dummy_model",
        model_id="dummy_model",
        head="universal",
        head_size=384,
    )
    index_config = IndexConfig(dimension=384, metric="cosine", hnsw_m=16)

    config = PipelineConfig(
        embedder_config=embedder_config,
        index_type=index_type,
        index_config=index_config,
        documents=documents,
        queries=queries,
        k=3,
    )

    return SearchPipeline(config)


if __name__ == "__main__":
    pipeline: SearchPipeline = create_sample_pipeline()
    result = pipeline.run_all()
    pipeline.print_results(result)
