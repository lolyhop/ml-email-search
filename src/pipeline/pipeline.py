import abc
import logging
import typing as tp

from src.retrievers.base import Embedder
from src.retrievers.dense import DenseEmbedder
from src.retrievers.dummy import DummyEmbedder
from src.index.entity import FaissEntity, FaissCorpus
from src.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class Pipeline(abc.ABC):
    """Pipeline for running end-to-end experiments."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config: PipelineConfig = config
        self.embedder: tp.Optional[Embedder] = None
        self.corpus: tp.Optional[FaissCorpus] = None
        self.build_time: float = 0.0

    def _create_embedder(self) -> Embedder:
        """Create and initialize embedder."""
        if self.config.embedder_config.model_name == "dummy_model":
            embedder = DummyEmbedder(self.config.embedder_config)
        else:
            embedder = DenseEmbedder(self.config.embedder_config)

        embedder.initialize()
        return embedder
    
    def _create_queries(self) -> tp.List[str]:
        """Create queries from config."""
        return self.config.queries

    def _create_corpus(self) -> FaissCorpus:
        """Create corpus from documents."""
        entities: tp.List[FaissEntity] = [
            FaissEntity(id=doc_id, content=content)
            for doc_id, content in self.config.documents
        ]
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

    @abc.abstractmethod
    def run_pipeline(self) -> tp.Any:
        """Run pipeline and return results."""
        pass
