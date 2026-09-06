import re


DESCRIPTIONS = {
    "UC_AdjustedResolutionParameters": "Calculates aligned base and upscaled image dimensions from width, height, scale, and multiple.",
    "UC_ResolutionSelectorExtended": "Calculates width and height from an aspect ratio and target megapixel count.",
    "UC_ImageScaleAndResolutionPicker": "Resizes an image to a megapixel target and returns base and upscaled dimensions.",
    "UC_Image_Color_Noise": "Adds configurable color noise to an image.",
    "UC_LoadImagePath": "Loads an image and mask from an explicit filesystem path.",
    "UC_LoadImageDirectory": "Loads images from a directory for batch or iterative workflows.",
    "UC_LoadImageWithAlpha": "Loads an image like Core Load Image and also returns an RGBA IMAGE with its alpha preserved.",
    "UC_ListToImageBatch": "Combines a list of compatible images into one image batch.",
    "UC_ImageMatchProperties": "Adjusts an image to match the size and properties of a reference image.",
    "UC_OpticalFlowComposite": "Composites images using motion estimated with optical flow.",
    "UC_ImageInwardEdgeFill": "Fills image edges inward to extend surrounding content.",
    "UC_ImageIterativeStretchFill": "Fills image borders by iteratively stretching nearby pixels.",
    "UC_TextOverlayNode": "Draws configurable text over an image.",
    "UC_ModifyMask": "Expands, contracts, blurs, or otherwise adjusts a mask.",
    "UC_SAM31CheckpointLoader": "Loads SAM 3.1 with automatic or deterministic FP32 precision.",
    "UC_MaskToBoundingBox": "Returns the smallest Core-compatible bounding box containing the mask.",
    "UC_ImageBlendByMask": "Blends two images using a mask and configurable blend behavior.",
    "UC_ImagePad": "Pads an image and produces the corresponding placement mask.",
    "UC_NoHaloLoHaloDownscale": "Downscales an image to a multiple-aligned megapixel target with NoHalo or LoHalo interpolation.",
    "UC_CropByMask": "Crops an image and mask to padded, dimension-aligned nonzero mask bounds.",
    "UC_ImageCropMerge": "Merges a processed crop back into its original image region.",
    "UC_ImageAndMaskResize": "Resizes an image and mask together to explicit or reference dimensions.",
    "UC_ResizeMask": "Resizes a mask with selectable interpolation and crop behavior.",
    "UC_BackgroundRemovalPreserveAlpha": "Removes an image background while preserving source resolution and soft alpha.",
    "UC_UnifiedBackgroundReplace": "Removes backgrounds from multiple foreground images and independently places each over one shared background.",
    "UC_StagedLayeredBackgroundComposite": "Places staged foreground cutouts over one background without rerunning background removal.",
    "UC_StagedLayerCrops": "Crops selected staged compositor layers from the composed image as an ordered image list.",
    "UC_StagedIndividualComposites": "Interactively stages foregrounds and renders each one independently over the background.",
    "UC_ExtractImage": "Extracts one image by index from a native ComfyUI image list.",
    "UC_StagedLayeredBackgroundCompositeOptions": "Configures reusable staged background-compositing cleanup and resize settings.",
    "UC_StagedMediaPipeFaceOptions": "Configures face detection, extraction, feathering, and initial placement.",
    "UC_StagedMediaPipeFaceBackgroundComposite": "Stages foregrounds and independently placeable MediaPipe face layers.",
    "UC_LayeredBackgroundComposite": "Builds one scene by removing and independently placing foreground objects in socket order.",
    "UC_MediaPipeFaceCompositeOptions": "Configures landmark regions and blending options for MediaPipe face compositing.",
    "UC_MediaPipeFaceComposite": "Detects face landmarks and composites selected facial regions between images.",
    "UC_SystemMessagePresets": "Provides reusable system-message presets for image prompting.",
    "UC_SystemMessageVideoPresets": "Provides reusable system-message presets for video prompting.",
    "UC_InstructPromptPresets": "Provides reusable image instruction prompt presets.",
    "UC_InstructPromptVideoPresets": "Provides reusable video instruction prompt presets.",
    "UC_BonusPromptPresets": "Provides optional prompt enhancement presets for images.",
    "UC_BonusPromptVideoPresets": "Provides optional prompt enhancement presets for videos.",
    "UC_LegacyPromptPresets": "Provides older prompt presets retained for workflow compatibility.",
    "UC_EditTargetPresets": "Provides preset descriptions of image regions or subjects to edit.",
    "UC_EditOpPresets": "Provides preset image editing operations and instructions.",
    "UC_CameraShotPresets": "Provides camera framing, angle, and shot presets.",
    "UC_UnifiedPresets": "Combines multiple prompt preset families into one selector.",
    "UC_VLMSysInstrPresets": "Provides system instruction presets for vision-language models.",
    "UC_VLMSysInstrPresetsExperimental": "Provides experimental system instruction presets for vision-language models.",
    "UC_VLMSysInstrLegacyPresets": "Provides legacy vision-language system instruction presets for compatibility workflows.",
    "UC_VLMSysQueryAddPresets": "Provides supplemental query presets for vision-language models.",
    "UC_VLMSysQueryRawPresets": "Outputs vision-language query instructions without request-wrapper syntax.",
    "UC_VLMSysInstrAdvPresets": "Provides advanced system instruction presets for vision-language models.",
    "UC_VLMSysInstrAdvPresetsExperimental": "Provides experimental VLM system instruction presets with query and override controls.",
    "UC_MiniMaxH3VLMSysInstrPresets": "Provides dedicated Base, First/Last Frame, and Reference VLM instructions for MiniMax H3.",
    "UC_MiniMaxH3VLMSysInstrPresetsExperimental": "Provides experimental system instruction presets for MiniMax H3 video workflows.",
    "UC_MiniMaxH3VLMSysInstrAdvPresets": "Provides MiniMax H3 VLM instructions with explicit user and system overrides.",
    "UC_MiniMaxH3VLMSysInstrAdvPresetsExperimental": "Provides experimental MiniMax H3 instructions with explicit user and system overrides.",
    "UC_AttentionBiasTextEncode": "Encodes text while applying token-level attention bias controls.",
    "UC_EmbeddingDetokenizerAnalysis": "Analyzes stored embedding vectors against every compatible connected text-encoder vocabulary.",
    "UC_TextConsensusBlendConfig": "Configures consensus-based blending of multiple text conditioning tensors.",
    "UC_VisualFusionConfig": "Configures grid-aware fusion of visual token embeddings from multiple images.",
    "UC_AdvancedVisualConditioningEncodeTokenFusion": "Fuses per-image visual tokens and DeepStack inputs before one shared conditioning encode.",
    "UC_Krea2TokenAttentionWeightTokenFusion": "Applies Krea2 attention weighting after token-first visual fusion and one shared conditioning encode.",
    "UC_AdvMiniMaxH3ImageToVideoTokenFusion": "Builds MiniMax H3 conditioning by fusing Picture tokens before one shared Qwen encode.",
    "UC_MiniMaxH3VLMGuide": "Experimentally inserts independently encoded timestamp and image conditioning before an H3 prompt.",
    "UC_AdvMiniMaxH3ImageToVideoTemporalFusion": "Experimentally fuses corresponding video conditioning blocks from separate Qwen encodes into lane zero.",
    "UC_AdvMiniMaxH3ImageToVideoTemporalTokenFusion": "Experimentally fuses corresponding video tokens and DeepStack features before a shared Qwen encode.",
    "UC_ConditioningConsensusBlend": "Blends multiple conditioning outputs after encoding while preserving reference placement.",
    "UC_TextEncodeLtxv2SystemPrompt": "Encodes LTXV2 text with a custom system prompt.",
    "UC_TextEncodeSystemPrompt": "Encodes text with a custom system prompt using the connected text encoder.",
    "UC_WeightedTextEncodeSystemPrompt": "Encodes weighted text with a custom system prompt.",
    "UC_TextEncodeSystemEditAdvanced": "Encodes advanced image-edit conditioning with custom prompts and multiple images.",
    "UC_TextEncodeGemmaSystemEditAdvanced": "Encodes advanced Gemma image-edit conditioning with custom system prompts.",
    "UC_AdvancedVisualConditioningEncode": "Encodes and fuses visual conditioning from multiple images with advanced controls.",
    "UC_MiniMaxH3MediaConfig": "Packages timestamped images as one Qwen Video sequence with optional native MiniMax H3 audio conditioning.",
    "UC_MiniMaxH3RadialAttentionConfig": "Configures sparse radial attention for target video rows in MiniMax H3.",
    "UC_UnifiedAttentionPatcher": "Applies one selected attention implementation to a cloned model.",
    "UC_AdvancedVisConEncoder": "Spatially fuses visual sources at each encoder resolution, then consensus-blends the complete conditionings.",
    "UC_AdvancedVisConEncoderTokenFusion": "Fuses per-source visual tokens at each encoder resolution before consensus-blending the completed resolution samples.",
    "UC_VisualConsensusConfiguration": "Configures independent spatial-fusion and complete-conditioning consensus stages.",
    "UC_AdvancedConsensusConfiguration": "Extends Text Consensus Blend Configurator with resolution count and offset controls.",
    "UC_VLMInputEmbeds": "Exports raw input embeddings from a supported vision-language model encoder.",
    "UC_Krea2TokenAttentionWeight": "Applies phrase-level attention weights while encoding Krea2 visual conditioning.",
    "UC_TextGenerate": "Generates text from images and timestamped Qwen3-VL video frames, with optional visual-token fusion.",
    "UC_TextGenerateQwen35SystemPrompt": "Generates Qwen3.5 text with optional image input and a custom system message.",
    "UC_SwitchInverseNode": "Selects between two inputs using an inverted boolean switch.",
    "UC_SoftSwitchInverseNode": "Selects available inputs using an inverted soft switch.",
    "UC_IntegerRangeRandom": "Returns a seeded random integer within a configurable range.",
    "UC_SeedCluster": "Generates a main seed and an incremented list of eight seeds.",
    "UC_FromSeedCluster": "Unpacks a seed cluster into eight reusable integer outputs.",
    "UC_GetJsonValue": "Selects a typed value from a top-level JSON object by key, index, or seed.",
    "UC_TagNormalizeCombine": "Normalizes, deduplicates, scores, and combines tag strings.",
    "UC_RandInt": "Outputs an integer widget with generation-time control behavior.",
    "UC_StaticInt": "Outputs one reusable static integer value.",
    "UC_StaticFloat": "Outputs one reusable static floating-point value with 0.01 increments.",
    "UC_RandIntRange": "Returns a deterministic random integer between minimum and maximum values.",
    "UC_ColorConvertNode": "Converts colors between picker, hexadecimal, integer, and RGB string formats.",
    "UC_ExtractBoundingBox": "Extracts x, y, width, and height from a bounding-box value.",
    "UC_AdjustBoundingBox": "Expands and aligns a selected bounding box around its center.",
    "UC_ExtractMask": "Selects one mask by index from a mask batch.",
    "UC_Ideogram4BoundingBoxCrop": "Crops an image to a selected box and returns the clamped box plus Ideogram-normalized coordinates.",
    "UC_Krea2LayerProbe": "Measures and optionally saves Krea2 conditioning-layer activation statistics.",
    "UC_Krea2LayerAblator": "Removes selected refusal-direction components from Krea2 conditioning layers.",
    "UC_EncoderNodesGuide": "Returns documentation for the node pack's advanced encoder workflows.",
    "UC_CompositeNodesGuide": "Returns documentation for background replacement, staged compositing, and MediaPipe face workflows.",
    "UC_HighResolutionTilingGuide": "Returns documentation for splitting, sampling, and recombining high-resolution image tiles.",
    "UC_LoraLoaderCLIPOnly": "Loads a LoRA into the text encoder without modifying the diffusion model.",
    "UC_BoldFrakturTextStyle": "Converts supported text characters to bold Fraktur Unicode styling.",
    "UC_UnBoldFrakturTextStyle": "Converts bold Fraktur Unicode characters back to plain text.",
    "UC_WordJoiner": "Joins words with Unicode word-joiner characters.",
    "UC_UnWordJoiner": "Removes Unicode word-joiner characters from text.",
    "UC_JSONMinifyRepair": "Repairs common JSON formatting issues and returns compact JSON text.",
    "UC_StringUnescape": "Converts escaped character sequences into their literal string values.",
    "UC_TextConcatenateAutogrow": "Converts ordered wildcard inputs to text and joins them with a connected delimiter.",
    "UC_TextConcatenateListsAutogrow": "Joins index-aligned text lists, broadcasting scalar values and applying scalar or per-index delimiters.",
    "UC_Newline": "Outputs one newline character.",
    "UC_MathAdd": "Adds two or more numbers from left to right.",
    "UC_MathSubtract": "Subtracts each following number from the first, from left to right.",
    "UC_MathMultiply": "Multiplies two or more numbers from left to right.",
    "UC_MathDivide": "Divides from left to right and can return zero instead of failing when a divisor is zero.",
    "UC_MathPower": "Raises the base number to the selected exponent, such as squaring with an exponent of 2.",
    "UC_MathFloor": "Rounds a number down to the nearest whole number.",
    "UC_MathCeil": "Rounds a number up to the nearest whole number.",
    "UC_MathRound": "Rounds a number to the selected number of decimal places.",
    "UC_MathModulo": "Returns the remainder after division, or zero when the divisor is zero.",
    "UC_MathAbs": "Returns a number's distance from zero, removing any negative sign.",
    "UC_MathSqrt": "Returns the square root of a number, or zero for a negative input.",
    "UC_MathSin": "Calculates the sine of an angle supplied in radians or degrees.",
    "UC_MathCos": "Calculates the cosine of an angle supplied in radians or degrees.",
    "UC_MathTan": "Calculates the tangent of an angle supplied in radians or degrees.",
    "UC_MathMin": "Returns the smallest of the connected numbers.",
    "UC_MathMax": "Returns the largest of the connected numbers.",
    "UC_MathClamp": "Keeps a number within a minimum and maximum by replacing values outside that range with the nearest limit.",
    "UC_MathNumberConvert": "Converts one number into both integer and floating-point outputs.",
    "UC_StringToNumber": "Converts numeric text to a number and returns the connected default when conversion fails.",
    "UC_NumberToString": "Converts an integer or floating-point number to text.",
    "UC_MathCompare": "Compares two numbers using the selected rule and returns true or false.",
    "UC_MathOperation": "Applies the selected basic operation: add, subtract, multiply, or divide, to two values.",
    "UC_MathAspectRatio": "Reduces width and height to their simplest whole-number aspect ratio.",
    "UC_LogicIF": "Returns the true value when the condition is true, otherwise the optional false value.",
    "UC_LogicAND": "Returns true only when every connected input is true.",
    "UC_LogicOR": "Returns true when at least one connected input is true.",
    "UC_LogicNOT": "Reverses a boolean value: true becomes false and false becomes true.",
    "UC_LogicXOR": "Returns true when an odd number of connected inputs are true.",
    "UC_SigmoidOffsetScheduler": "Creates a curved denoising sigma schedule with adjustable steepness and early/late emphasis; originally made for the Chroma model.",
    "Ideogram4SchedulerPreset": "Provides scheduler and sampling parameters tuned for Ideogram 4 workflows.",
}


EXTRA_ALIASES = {
    "UC_TextConcatenateAutogrow": ["concatenate", "text concat", "join text", "merge text", "combine strings", "autogrow"],
    "UC_Newline": ["newline", "new line", "line break", r"\n"],
    "UC_StaticInt": ["primitive integer", "number", "shared integer", "constant int"],
    "UC_StaticFloat": ["primitive float", "number", "decimal", "megapixel", "shared value", "constant float"],
    "UC_GetJsonValue": ["json", "value", "random value", "key value", "configuration"],
    "UC_SeedCluster": ["seed list", "incremented seeds", "random seed"],
    "UC_FromSeedCluster": ["unpack seeds", "seed list", "distribute seeds"],
    "UC_ConditioningConsensusBlend": ["conditioning merge", "conditioning combine", "post encoder", "cwb"],
    "UC_VisualFusionConfig": ["image token fusion", "visual blend", "dither", "checkerboard"],
    "UC_AdvancedVisualConditioningEncodeTokenFusion": ["token fusion", "visual token encode"],
    "UC_Krea2TokenAttentionWeightTokenFusion": ["token fusion", "Krea2 attention"],
    "UC_AdvMiniMaxH3ImageToVideoTokenFusion": ["token fusion", "MiniMax H3"],
    "UC_AdvancedVisConEncoder": ["visual consensus", "resolution consensus", "deepstack fusion"],
    "UC_AdvancedVisConEncoderTokenFusion": ["visual consensus", "token fusion", "resolution consensus"],
    "UC_VisualConsensusConfiguration": ["visual consensus config", "spatial consensus"],
    "UC_TextConsensusBlendConfig": ["conditioning blend config", "text merge", "cwb config"],
    "UC_MediaPipeFaceComposite": ["face swap", "face landmarks", "face blend", "mediapipe"],
    "UC_MediaPipeFaceCompositeOptions": ["face regions", "face landmarks", "mediapipe options"],
    "UC_UnifiedBackgroundReplace": ["background replacement", "remove background", "batch composite", "shared background", "cutout"],
    "UC_BackgroundRemovalPreserveAlpha": ["remove background", "rgba", "transparent foreground", "alpha mask"],
    "UC_StagedLayeredBackgroundComposite": ["staged scene composite", "interactive placement", "cached cutout"],
    "UC_StagedIndividualComposites": ["staged separate composites", "one foreground per image", "interactive placement"],
    "UC_LayeredBackgroundComposite": ["scene composite", "layered composite", "object placement", "background replacement", "cutout"],
    "UC_ResolutionSelectorExtended": ["megapixels", "aspect ratio", "width height", "resolution"],
    "UC_ImageScaleAndResolutionPicker": ["megapixels", "image resize", "upscale", "resolution"],
    "UC_LoadImagePath": ["load image", "image path", "absolute path"],
    "UC_LoadImageDirectory": ["image folder", "batch loader", "directory loader"],
    "UC_LoadImageWithAlpha": ["load image", "alpha", "rgba", "transparent png"],
    "UC_TextGenerate": ["llm", "vlm", "chat", "text generation", "token fusion", "video"],
    "UC_TextGenerateQwen35SystemPrompt": ["llm", "vlm", "qwen", "chat", "system prompt"],
    "UC_VLMInputEmbeds": ["embedding export", "visual embeddings", "qwen embeddings", "krea embeddings"],
    "UC_LoraLoaderCLIPOnly": ["clip lora", "text encoder lora", "load lora"],
}


def _search_terms(schema):
    text = f"{schema.node_id} {schema.display_name or ''} {schema.category or ''}"
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+", text)
    ignored = {"uc", "node", "utils", "utilities", "advanced"}
    terms = [word.lower() for word in words if word.lower() not in ignored]
    display = (schema.display_name or "").strip().lower()
    if display:
        terms.append(display)
    return terms


def enrich_node_metadata(node_class):
    if node_class.__dict__.get("_uc_metadata_enriched", False):
        return
    original = node_class.define_schema.__func__

    @classmethod
    def define_schema(cls):
        schema = original(cls)
        if cls is not node_class or schema.is_deprecated or "(Legacy)" in (schema.display_name or ""):
            return schema
        description = DESCRIPTIONS.get(schema.node_id)
        if not schema.description:
            schema.description = description or f"Provides {schema.display_name or schema.node_id} functionality."
        aliases = list(schema.search_aliases or [])
        aliases.extend(_search_terms(schema))
        aliases.extend(EXTRA_ALIASES.get(schema.node_id, []))
        schema.search_aliases = list(dict.fromkeys(alias for alias in aliases if alias))
        return schema

    node_class.define_schema = define_schema
    node_class._uc_metadata_enriched = True


def enrich_node_list(node_classes):
    for node_class in node_classes:
        enrich_node_metadata(node_class)
    return node_classes
