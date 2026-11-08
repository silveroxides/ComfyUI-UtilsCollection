from comfy_api.latest import io
from folder_paths import get_filename_list, get_full_path_or_raise

from ..helpers.embedding_helpers import analyze_embedding_file


class UC_EmbeddingDetokenizerAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_EmbeddingDetokenizerAnalysis",
            display_name="Embedding Detokenizer Analysis",
            category="advanced/text",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input("embedding_name", options=get_filename_list("embeddings"), tooltip="Select an embedding from ComfyUI's embeddings directory."),
                io.Int.Input("top_k", default=5, min=1, max=100, step=1, tooltip="How many matching words to show for each embedding."),
                io.Combo.Input("similarity_metric", options=["cosine", "euclidean", "dot_product"], default="cosine"),
                io.Int.Input("alignment_candidates", default=64, min=8, max=512, step=8, tooltip="How many possible word matches to consider for each embedding."),
                io.Boolean.Input("latin_only", default=False, tooltip="Show only Latin text and common punctuation in word matches."),
            ],
            outputs=[io.String.Output(display_name="analysis"), io.String.Output(display_name="approximation"), io.String.Output(display_name="cwb_approximation")],
        )

    @classmethod
    def execute(cls, clip, embedding_name: str, top_k: int, similarity_metric: str, alignment_candidates: int = 64, latin_only: bool = False) -> io.NodeOutput:
        embedding_path = get_full_path_or_raise("embeddings", embedding_name)
        analysis, approximation, cwb_approximation = analyze_embedding_file(clip, embedding_path, top_k, similarity_metric, alignment_candidates, latin_only)
        return io.NodeOutput(analysis, approximation, cwb_approximation)
