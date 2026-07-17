from xidootopic import TextCleaner,TextChunker,TextEmbedder,HFencoder,MeanPooling,UMAPReducer,IsolationForestOutlierRemover,KNNGraphBuilder,LeidenResolutionOptimizer,LeidenClusterer,ClusterFilter,ClusterMerger,KeyWordExtractor,RepresentativeDocsExtractor
class XidooTopic:
    @classmethod
    def default(cls, verbose=True):
        return cls(
            preprocessor=TextCleaner(),
            chunker=TextChunker(),
            embedder=TextEmbedder.default(),
            reducer=UMAPReducer(),
            outlier_detector=IsolationForestOutlierRemover(),
            graph_builder=KNNGraphBuilder(),
            optimal_resolution=LeidenResolutionOptimizer(),
            clusterer=LeidenClusterer(),
            cluster_filter=ClusterFilter(),
            cluster_merger=ClusterMerger(),
            keyword_model=KeyWordExtractor.default(),
            doc_model=RepresentativeDocsExtractor(),
            verbose=verbose
        )
    def __init__(self,
                 preprocessor,
                 chunker,
                 embedder,
                 reducer,
                 outlier_detector,
                 graph_builder,
                 optimal_resolution,
                 clusterer,
                 cluster_filter,
                 cluster_merger,
                 keyword_model,
                 doc_model,
                 verbose=True) :
        self.preprocessor = preprocessor
        self.chunker=chunker 
        self.embedder = embedder
        self.reducer = reducer
        self.outlier_detector = outlier_detector
        self.graph_builder = graph_builder
        self.optimal_resolution=optimal_resolution
        self.clusterer = clusterer
        self.cluster_filter = cluster_filter
        self.cluster_merger = cluster_merger
        self.keyword_model = keyword_model
        self.doc_model = doc_model
        self.verbose = verbose
        pass
    def fit_transform(self,texts):
        texts_clean,ids=self.preprocessor.transform(texts)
        texts_chunked=self.chunker.recortaTodo(texts_clean)
        embeddings=self.embedder.embed_texts(texts_chunked)
        emb_red = self.reducer.fit_transform(embeddings)
        out = self.outlier_detector.fit_transform(emb_red, texts_chunked)
        emb_f=out["embeddings"]
        texts_f=out.get("texts",None)
        graph = self.graph_builder.fit_transform(emb_f)
        _,res=self.optimal_resolution.fit_predict(graph,emb_f) 
        labels = self.clusterer.fit_predict(graph)
        emb_f2, texts_f2, labels_f2, _, _ = self.cluster_filter.fit_transform(emb_f, texts_f, labels)
        labels_final = self.cluster_merger.fit(emb_f2, labels_f2)
        topics = self.keyword_model.extract(
            texts_f2, emb_f2, labels_final
        )
        # 10. Representative docs
        docs = self.doc_model.extract(
            emb_f2, labels_final, texts_f2
        )

        return {
            "topics": topics,
            "documents": docs,
            "embeddings": emb_f2,
            "labels": labels_final
        }
