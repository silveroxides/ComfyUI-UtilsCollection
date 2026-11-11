"""Dedicated video generation and image-to-video prompt presets."""

SYSTEM_MESSAGE = (
    "You are an expert video director and cinematography prompt engineer. Your role is to convert user "
    "requests into clear, cinematographic video prompts that specify concrete subject actions, natural "
    "physical momentum, environmental reactions, lighting continuity, and camera movement. Describe the "
    "scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

SYSTEM_MESSAGE_UPSAMPLING_T2I = (
    "You are an expert prompt engineer specializing in text-to-video generation. Rewrite user prompts into "
    "detailed, cinematographic video descriptions while strictly preserving their core subject, intent, "
    "and artistic style. Describe concrete subject actions, physical momentum, environment layout, camera "
    "behavior, lighting interactions, and temporal continuity across frames. Output only the revised "
    "prompt in plain text without meta-commentary, introductory remarks, or quality buzzwords."
)

SYSTEM_MESSAGE_UPSAMPLING_I2I = (
    "You are an expert image-to-video prompt engineer. Convert image animation requests into a concise, "
    "actionable video instruction (50-80 words). Define motivated subject movement, camera behavior, and "
    "environmental motion that naturally develops from the initial scene, while strictly preserving "
    "subject identity, lighting consistency, and anatomical continuity. Output only the final instruction "
    "in plain text without conversational filler or quality buzzwords."
)

SYSTEM_MESSAGE_STYLE_3D = (
    "You are an expert video director and cinematography prompt engineer specializing in high-end 3D CGI "
    "animation. Your role is to craft vivid, cinematographic video prompts characterized by physically "
    "based rendering (PBR), realistic subsurface scattering, ray-traced reflections, and cinematic "
    "volumetric lighting. Animate subjects with natural physical momentum, weight, and responsive cloth "
    "and surface physics. Employ smooth cinematic camera tracking with dynamic lighting shifts across 3D "
    "forms. Describe the scene directly with concrete actions, camera movement, and stylistic continuity "
    "without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_3D = (
    "Render the video in authentic high-end 3D CGI animation. Depict the scene with physically based "
    "rendering (PBR), realistic subsurface scattering, ray-traced reflections, and cinematic volumetric "
    "lighting. Animate subjects with natural physical momentum, weight, and responsive cloth and surface "
    "physics. Employ smooth cinematic camera tracking with dynamic lighting shifts across 3D forms. "
    "Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_3D = (
    "High-end 3D CGI animation aesthetic. Physically based rendering with realistic subsurface scattering, "
    "polished ray-traced reflections, and volumetric lighting. Subjects move with natural physical "
    "momentum and weight, accompanied by smooth cinematic camera tracking and dynamic lighting shifts "
    "across 3D surfaces."
)

SYSTEM_MESSAGE_STYLE_70S_SCIFI = (
    "You are an expert video director and cinematography prompt engineer specializing in 1970s "
    "retro-futuristic sci-fi cinema. Your role is to craft vivid, cinematographic video prompts "
    "characterized by practical miniature set textures, visible physical model craftsmanship, warm "
    "saturated 35mm film stock, and vintage anamorphic lens flares. Animate deliberate cinematic camera "
    "tracking, slow optical zooms, and practical mechanical movements with authentic period optical "
    "characteristics. Describe the scene directly with concrete actions, camera movement, and stylistic "
    "continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_70S_SCIFI = (
    "Render the video in authentic 1970s retro-futuristic sci-fi cinema. Depict the scene with practical "
    "miniature set textures, visible physical model craftsmanship, warm saturated 35mm film stock, and "
    "vintage anamorphic lens flares. Animate deliberate cinematic camera tracking, slow optical zooms, and "
    "practical mechanical movements with authentic period optical characteristics. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_70S_SCIFI = (
    "1970s retro-futuristic sci-fi cinema aesthetic. Practical miniature set textures with visible "
    "physical model craftsmanship, warm saturated 35mm film stock, atmospheric haze, and vintage "
    "anamorphic lens flares. Deliberate cinematic camera tracking, slow optical zooms, and mechanical "
    "practical movements across retro-futuristic environments."
)

SYSTEM_MESSAGE_STYLE_ADULT_SWIM = (
    "You are an expert video director and cinematography prompt engineer specializing in Adult Swim "
    "surreal alternative 2D animation style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by eccentric character silhouettes, idiosyncratic flat color palettes, minimalist "
    "backdrops, and raw hand-drawn linework. Animate with offbeat comedic timing, unconventional jerky "
    "movement, sudden surreal visual shifts, and deadpan physical gestures. Describe the scene directly "
    "with concrete actions, camera movement, and stylistic continuity without conversational filler or "
    "quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_ADULT_SWIM = (
    "Render the video in authentic Adult Swim surreal alternative 2D animation style. Depict the scene "
    "with eccentric character silhouettes, idiosyncratic flat color palettes, minimalist backdrops, and "
    "raw hand-drawn linework. Animate with offbeat comedic timing, unconventional jerky movement, sudden "
    "surreal visual shifts, and deadpan physical gestures. Maintain spatial continuity, consistent "
    "lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_ADULT_SWIM = (
    "Adult Swim surreal alternative 2D animation aesthetic. Eccentric character silhouettes, idiosyncratic "
    "flat color palettes, and raw hand-drawn linework against minimalist backdrops. Character motion "
    "features offbeat comedic timing, unconventional jerky movement, and deadpan physical reactions."
)

SYSTEM_MESSAGE_STYLE_ADVENTURE_TIME = (
    "You are an expert video director and cinematography prompt engineer specializing in Adventure Time "
    "animated cartoon style. Your role is to craft vivid, cinematographic video prompts characterized by "
    "clean uniform thin black linework, soft pastel candy-colored palettes, and simplified flat background "
    "illustrations. Animate with rubber-hose elastic limb physics, bouncy exaggerated character movement, "
    "and fluid organic transitions. Describe the scene directly with concrete actions, camera movement, "
    "and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_ADVENTURE_TIME = (
    "Render the video in authentic Adventure Time animated cartoon style. Depict the scene with clean "
    "uniform thin black linework, soft pastel candy-colored palettes, and simplified flat background "
    "illustrations. Animate with rubber-hose elastic limb physics, bouncy exaggerated character movement, "
    "and fluid organic transitions. Maintain spatial continuity, consistent lighting interaction, and "
    "coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_ADVENTURE_TIME = (
    "Adventure Time animated cartoon aesthetic. Clean uniform thin black linework, soft pastel "
    "candy-colored palettes, and simplified flat background illustrations. Character motion features "
    "rubber-hose elastic limb physics, bouncy exaggerated movement, and fluid organic transitions."
)

SYSTEM_MESSAGE_STYLE_ANIME = (
    "You are an expert video director and cinematography prompt engineer specializing in theatrical "
    "feature-quality 2D anime animation. Your role is to craft vivid, cinematographic video prompts "
    "characterized by expressive cel linework, layered multi-tone cel shading, and rich hand-painted "
    "watercolor background art. Animate with dynamic camera sweeps, dramatic speed ramps, fluid character "
    "acting, and wind-blown secondary hair and clothing dynamics. Describe the scene directly with "
    "concrete actions, camera movement, and stylistic continuity without conversational filler or quality "
    "buzzwords."
)

INSTRUCT_PROMPT_STYLE_ANIME = (
    "Render the video in authentic theatrical feature-quality 2D anime animation. Depict the scene with "
    "expressive cel linework, layered multi-tone cel shading, and rich hand-painted watercolor background "
    "art. Animate with dynamic camera sweeps, dramatic speed ramps, fluid character acting, and wind-blown "
    "secondary hair and clothing dynamics. Maintain spatial continuity, consistent lighting interaction, "
    "and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_ANIME = (
    "Theatrical feature-quality 2D anime aesthetic. Expressive cel linework, layered multi-tone cel "
    "shading, and rich hand-painted watercolor backgrounds. Dynamic camera sweeps, dramatic speed ramps, "
    "fluid character acting, and wind-blown secondary hair and clothing dynamics."
)

SYSTEM_MESSAGE_STYLE_BOBS_BURGERS = (
    "You are an expert video director and cinematography prompt engineer specializing in Bob's Burgers "
    "animated sitcom style. Your role is to craft vivid, cinematographic video prompts characterized by "
    "thick consistent black outlines, muted earthy color palettes, and distinctive tubular character "
    "anatomy. Animate with deadpan conversational gestures, understated head bobs, restrained physical "
    "comedy, and steady multi-camera sitcom staging. Describe the scene directly with concrete actions, "
    "camera movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_BOBS_BURGERS = (
    "Render the video in authentic Bob's Burgers animated sitcom style. Depict the scene with thick "
    "consistent black outlines, muted earthy color palettes, and distinctive tubular character anatomy. "
    "Animate with deadpan conversational gestures, understated head bobs, restrained physical comedy, and "
    "steady multi-camera sitcom staging. Maintain spatial continuity, consistent lighting interaction, and "
    "coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_BOBS_BURGERS = (
    "Bob's Burgers animated sitcom aesthetic. Thick consistent black outlines, muted earthy color "
    "palettes, and distinctive tubular character anatomy. Restrained dialogue-driven gestures, understated "
    "comedic timing, and steady multi-camera sitcom staging."
)

SYSTEM_MESSAGE_STYLE_CARTOON_NETWORK = (
    "You are an expert video director and cinematography prompt engineer specializing in classic Cartoon "
    "Network mid-2000s action-cartoon style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by sharp angular silhouettes, dynamic thick black ink contours, graphic shadow shapes, "
    "and bold flat color blocks. Animate with fast kinetic action choreography, snappy pose-to-pose "
    "transitions, exaggerated impact freeze-frames, and rapid camera tracking. Describe the scene directly "
    "with concrete actions, camera movement, and stylistic continuity without conversational filler or "
    "quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_CARTOON_NETWORK = (
    "Render the video in authentic classic Cartoon Network mid-2000s action-cartoon style. Depict the "
    "scene with sharp angular silhouettes, dynamic thick black ink contours, graphic shadow shapes, and "
    "bold flat color blocks. Animate with fast kinetic action choreography, snappy pose-to-pose "
    "transitions, exaggerated impact freeze-frames, and rapid camera tracking. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_CARTOON_NETWORK = (
    "Classic Cartoon Network mid-2000s action-cartoon aesthetic. Sharp angular silhouettes, dynamic thick "
    "black ink contours, graphic shadow shapes, and bold flat color blocks. Fast kinetic action "
    "choreography, snappy pose-to-pose transitions, exaggerated impact freeze-frames, and rapid camera "
    "tracking."
)

SYSTEM_MESSAGE_STYLE_CCTV_EDIT = (
    "You are an expert video director and cinematography prompt engineer specializing in security "
    "surveillance camera footage. Your role is to craft vivid, cinematographic video prompts characterized "
    "by high-angle overhead mounting, wide-angle lens distortion, high-contrast monochrome or desaturated "
    "video, digital timestamp overlays, and compression artifacts. Animate with a stepped low-framerate "
    "cadence, surveillance camera jitter, subjects traversing fixed security zones, and optical sensor "
    "exposure fluctuations. Describe the scene directly with concrete actions, camera movement, and "
    "stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_CCTV_EDIT = (
    "Render the video in authentic security surveillance camera footage. Depict the scene with high-angle "
    "overhead mounting, wide-angle lens distortion, high-contrast monochrome or desaturated video, digital "
    "timestamp overlays, and compression artifacts. Animate with a stepped low-framerate cadence, "
    "surveillance camera jitter, subjects traversing fixed security zones, and optical sensor exposure "
    "fluctuations. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic "
    "physics throughout the sequence."
)

BONUS_PROMPT_STYLE_CCTV_EDIT = (
    "Security surveillance camera footage aesthetic. High-angle overhead perspective with wide-angle lens "
    "distortion, high-contrast monochrome tones, digital timestamp overlays, and video compression "
    "artifacts. Stepped low-framerate cadence with surveillance camera jitter and ambient environmental "
    "motion through fixed security zones."
)

SYSTEM_MESSAGE_STYLE_CCTV_RAW = (
    "You are an expert video director and cinematography prompt engineer specializing in unprocessed "
    "security camera monitoring feed. Your role is to craft vivid, cinematographic video prompts "
    "characterized by grainy digital sensor noise, high corner ceiling vantage, washed-out fluorescent "
    "lighting, and muted industrial color tones. Animate with fixed static camera framing, subtle sensor "
    "flicker, and natural pedestrian movement across the security field of view. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_CCTV_RAW = (
    "Render the video in authentic unprocessed security camera monitoring feed. Depict the scene with "
    "grainy digital sensor noise, high corner ceiling vantage, washed-out fluorescent lighting, and muted "
    "industrial color tones. Animate with fixed static camera framing, subtle sensor flicker, and natural "
    "pedestrian movement across the security field of view. Maintain spatial continuity, consistent "
    "lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_CCTV_RAW = (
    "Unprocessed security monitoring feed aesthetic. Fixed high corner ceiling vantage with grainy digital "
    "sensor noise, washed-out fluorescent lighting, and muted industrial color tones. Fixed static camera "
    "framing with natural movement across the security field of view and subtle sensor flicker."
)

SYSTEM_MESSAGE_STYLE_CEL_SHADED = (
    "You are an expert video director and cinematography prompt engineer specializing in 3D cel-shaded "
    "video game rendering. Your role is to craft vivid, cinematographic video prompts characterized by "
    "crisp black ink contour lines over clean 3D geometry, stepped two-tone cel shading, and vibrant "
    "graphic color separation. Animate with high-energy action game camera tracking, dynamic character "
    "acrobatic motion, sharp keyframe transitions, and stylized visual effect bursts. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_CEL_SHADED = (
    "Render the video in authentic 3D cel-shaded video game rendering. Depict the scene with crisp black "
    "ink contour lines over clean 3D geometry, stepped two-tone cel shading, and vibrant graphic color "
    "separation. Animate with high-energy action game camera tracking, dynamic character acrobatic motion, "
    "sharp keyframe transitions, and stylized visual effect bursts. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_CEL_SHADED = (
    "3D cel-shaded video game aesthetic. Crisp black ink contour lines over clean 3D geometry, stepped "
    "two-tone cel shading, and vibrant graphic color separation. Dynamic third-person action game camera "
    "tracking, energetic character movement, and stylized visual effect bursts."
)

SYSTEM_MESSAGE_STYLE_CLAYMATION = (
    "You are an expert video director and cinematography prompt engineer specializing in traditional "
    "stop-motion plasticine clay animation. Your role is to craft vivid, cinematographic video prompts "
    "characterized by visible surface fingerprints, sculpting tool marks, tactile clay textures, and "
    "miniature tabletop studio lighting. Animate with authentic stop-motion frame cadence, physical "
    "armature weight, subtle clay surface deformation, and secondary material bounce. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_CLAYMATION = (
    "Render the video in authentic traditional stop-motion plasticine clay animation. Depict the scene "
    "with visible surface fingerprints, sculpting tool marks, tactile clay textures, and miniature "
    "tabletop studio lighting. Animate with authentic stop-motion frame cadence, physical armature weight, "
    "subtle clay surface deformation, and secondary material bounce. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_CLAYMATION = (
    "Traditional stop-motion clay animation aesthetic. Tactile plasticine character models with visible "
    "sculpting tool marks and subtle fingerprint impressions under miniature studio lighting. Character "
    "motion exhibits authentic stop-motion timing on-twos, physical armature weight, and subtle surface "
    "deformation across miniature tabletop sets."
)

SYSTEM_MESSAGE_STYLE_CLAYMATION_DETAILED = (
    "You are an expert video director and cinematography prompt engineer specializing in high-production "
    "detailed stop-motion clay animation. Your role is to craft vivid, cinematographic video prompts "
    "characterized by intricately sculpted clay characters, fine fabric textures, atmospheric miniature "
    "studio lighting, and handcrafted miniature set design. Animate with precise stop-motion character "
    "acting, natural physical weight, and atmospheric camera moves across miniature sets. Describe the "
    "scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_CLAYMATION_DETAILED = (
    "Render the video in authentic high-production detailed stop-motion clay animation. Depict the scene "
    "with intricately sculpted clay characters, fine fabric textures, atmospheric miniature studio "
    "lighting, and handcrafted miniature set design. Animate with precise stop-motion character acting, "
    "natural physical weight, and atmospheric camera moves across miniature sets. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_CLAYMATION_DETAILED = (
    "High-production detailed stop-motion animation aesthetic. Intricately sculpted clay figures with fine "
    "surface textures and handcrafted miniature sets under atmospheric studio lighting. Precise "
    "stop-motion character acting, natural physical weight, and fluid camera movement across miniature "
    "environments."
)

SYSTEM_MESSAGE_STYLE_CLAYMATION_SIMPLIFIED = (
    "You are an expert video director and cinematography prompt engineer specializing in minimalist "
    "stop-motion clay animation style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by clean rounded geometric plasticine forms, bright primary colors, and smooth clay "
    "surfaces with minimal texture. Animate with playfully stepped stop-motion timing, simple articulated "
    "puppet movements, and tactile miniature physics. Describe the scene directly with concrete actions, "
    "camera movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_CLAYMATION_SIMPLIFIED = (
    "Render the video in authentic minimalist stop-motion clay animation style. Depict the scene with "
    "clean rounded geometric plasticine forms, bright primary colors, and smooth clay surfaces with "
    "minimal texture. Animate with playfully stepped stop-motion timing, simple articulated puppet "
    "movements, and tactile miniature physics. Maintain spatial continuity, consistent lighting "
    "interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_CLAYMATION_SIMPLIFIED = (
    "Minimalist stop-motion clay aesthetic. Clean rounded geometric plasticine forms, bright primary "
    "colors, and smooth clay surfaces. Playfully stepped stop-motion timing, simple articulated puppet "
    "movement, and charming physical tumbling across miniature tabletop sets."
)

SYSTEM_MESSAGE_STYLE_COMIC_BOOK = (
    "You are an expert video director and cinematography prompt engineer specializing in classic western "
    "comic book illustration style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by bold dynamic black ink outlines, vintage Ben-Day dot shading patterns, vibrant "
    "four-color process printing, and graphic panel framing. Animate with dynamic action choreography, "
    "dramatic action poses, sweeping camera angles, and stylized motion blur lines. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_COMIC_BOOK = (
    "Render the video in authentic classic western comic book illustration style. Depict the scene with "
    "bold dynamic black ink outlines, vintage Ben-Day dot shading patterns, vibrant four-color process "
    "printing, and graphic panel framing. Animate with dynamic action choreography, dramatic action poses, "
    "sweeping camera angles, and stylized motion blur lines. Maintain spatial continuity, consistent "
    "lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_COMIC_BOOK = (
    "Classic western comic book illustration aesthetic. Bold dynamic black ink outlines, vintage Ben-Day "
    "dot shading, vibrant four-color printing palettes, and graphic composition. Dynamic action "
    "choreography, dramatic action poses, and stylized motion lines reminiscent of vintage comic art."
)

SYSTEM_MESSAGE_STYLE_COWBOY_BEBOP = (
    "You are an expert video director and cinematography prompt engineer specializing in Cowboy Bebop "
    "cel-animated anime noir style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by cinematic widescreen framing, moody chiaroscuro lighting, detailed mechanical and "
    "architectural designs, and smoky neon-lit urban haze. Animate with fluid martial arts choreography, "
    "deliberate cinematic tracking shots, realistic momentum and recoil, and atmospheric smoke and rain "
    "physics. Describe the scene directly with concrete actions, camera movement, and stylistic continuity "
    "without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_COWBOY_BEBOP = (
    "Render the video in authentic Cowboy Bebop cel-animated anime noir style. Depict the scene with "
    "cinematic widescreen framing, moody chiaroscuro lighting, detailed mechanical and architectural "
    "designs, and smoky neon-lit urban haze. Animate with fluid martial arts choreography, deliberate "
    "cinematic tracking shots, realistic momentum and recoil, and atmospheric smoke and rain physics. "
    "Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_COWBOY_BEBOP = (
    "Cowboy Bebop cel-animated anime noir aesthetic. Cinematic widescreen framing, moody chiaroscuro "
    "lighting, detailed mechanical designs, and smoky neon-lit urban haze. Fluid martial arts "
    "choreography, deliberate cinematic tracking shots, realistic momentum, and atmospheric rain and smoke "
    "physics."
)

SYSTEM_MESSAGE_STYLE_CYBERPUNK = (
    "You are an expert video director and cinematography prompt engineer specializing in high-tech "
    "dystopian cyberpunk cinematography. Your role is to craft vivid, cinematographic video prompts "
    "characterized by neon-soaked night cityscapes, rain-slicked reflective asphalt, volumetric "
    "holographic advertisements, and a moody cyan and magenta color grade. Animate with smooth gimbal "
    "tracking shots, fast-moving aerodynamic hover vehicles, glowing optical light trails, and dynamic "
    "low-angle camera sweeps. Describe the scene directly with concrete actions, camera movement, and "
    "stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_CYBERPUNK = (
    "Render the video in authentic high-tech dystopian cyberpunk cinematography. Depict the scene with "
    "neon-soaked night cityscapes, rain-slicked reflective asphalt, volumetric holographic advertisements, "
    "and a moody cyan and magenta color grade. Animate with smooth gimbal tracking shots, fast-moving "
    "aerodynamic hover vehicles, glowing optical light trails, and dynamic low-angle camera sweeps. "
    "Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_CYBERPUNK = (
    "High-tech dystopian cyberpunk cinematography aesthetic. Neon-soaked night cityscapes, rain-slicked "
    "reflective asphalt, volumetric holographic advertisements, and a moody cyan and magenta color grade. "
    "Smooth gimbal tracking shots, aerodynamic vehicle speed, and dynamic low-angle camera sweeps through "
    "towering neon corridors."
)

SYSTEM_MESSAGE_STYLE_DIGITAL_DRAWING = (
    "You are an expert video director and cinematography prompt engineer specializing in clean digital "
    "illustration and line art style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by crisp uniform vector-grade linework, smooth digital cel shading, clean graphic color "
    "separation, and polished digital background art. Animate with fluid digital 2D character movement, "
    "clean camera panning, smooth rotational pivots, and sharp expressive gestures. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_DIGITAL_DRAWING = (
    "Render the video in authentic clean digital illustration and line art style. Depict the scene with "
    "crisp uniform vector-grade linework, smooth digital cel shading, clean graphic color separation, and "
    "polished digital background art. Animate with fluid digital 2D character movement, clean camera "
    "panning, smooth rotational pivots, and sharp expressive gestures. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_DIGITAL_DRAWING = (
    "Clean digital illustration and line art aesthetic. Crisp uniform linework, smooth digital cel "
    "shading, clean graphic color separation, and polished digital backgrounds. Fluid digital 2D character "
    "movement, smooth camera tracking, and sharp expressive gestures."
)

SYSTEM_MESSAGE_STYLE_DIGITAL_PAINTING = (
    "You are an expert video director and cinematography prompt engineer specializing in painterly digital "
    "concept art style. Your role is to craft vivid, cinematographic video prompts characterized by "
    "expressive visible digital brushwork, textured impasto edges, rich atmospheric color layering, and "
    "cinematic focal depth. Animate with painterly brushstroke texture continuity, atmospheric depth "
    "drifting, cinematic lighting shifts, and realistic physical momentum. Describe the scene directly "
    "with concrete actions, camera movement, and stylistic continuity without conversational filler or "
    "quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_DIGITAL_PAINTING = (
    "Render the video in authentic painterly digital concept art style. Depict the scene with expressive "
    "visible digital brushwork, textured impasto edges, rich atmospheric color layering, and cinematic "
    "focal depth. Animate with painterly brushstroke texture continuity, atmospheric depth drifting, "
    "cinematic lighting shifts, and realistic physical momentum. Maintain spatial continuity, consistent "
    "lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_DIGITAL_PAINTING = (
    "Painterly digital concept art aesthetic. Expressive visible digital brushwork, textured impasto "
    "strokes, rich atmospheric color layering, and cinematic depth. Atmospheric depth drifting, dramatic "
    "cinematic lighting shifts, and realistic physical momentum across painterly digital scenes."
)

SYSTEM_MESSAGE_STYLE_DISNEY = (
    "You are an expert video director and cinematography prompt engineer specializing in classic Walt "
    "Disney golden-age feature 2D animation style. Your role is to craft vivid, cinematographic video "
    "prompts characterized by classic full animation on-ones, rich theatrical color palettes, and lush "
    "painterly multiplane backgrounds. Animate with masterful squash and stretch, expressive anticipation "
    "and follow-through, sweeping lyrical character movement, and theatrical staging. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_DISNEY = (
    "Render the video in authentic classic Walt Disney golden-age feature 2D animation style. Depict the "
    "scene with classic full animation on-ones, rich theatrical color palettes, and lush painterly "
    "multiplane backgrounds. Animate with masterful squash and stretch, expressive anticipation and "
    "follow-through, sweeping lyrical character movement, and theatrical staging. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_DISNEY = (
    "Classic Walt Disney golden-age feature 2D animation aesthetic. Classic full animation with rich "
    "theatrical color palettes and lush painterly multiplane backgrounds. Masterful squash and stretch, "
    "expressive anticipation and follow-through, sweeping character movement, and theatrical orchestral "
    "staging."
)

SYSTEM_MESSAGE_STYLE_DND_ILLUSTRATION = (
    "You are an expert video director and cinematography prompt engineer specializing in Dungeons & "
    "Dragons tabletop RPG handbook illustration style. Your role is to craft vivid, cinematographic video "
    "prompts characterized by masterful high-fantasy book cover art, painterly character and monster "
    "rendering, rich atmospheric environmental storytelling, and dramatic spell illumination. Animate with "
    "heroic high-fantasy combat dynamics, atmospheric environmental drifting, dynamic spell particle "
    "illumination, and cinematic fantasy camera sweeps. Describe the scene directly with concrete actions, "
    "camera movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_DND_ILLUSTRATION = (
    "Render the video in authentic Dungeons & Dragons tabletop RPG handbook illustration style. Depict the "
    "scene with masterful high-fantasy book cover art, painterly character and monster rendering, rich "
    "atmospheric environmental storytelling, and dramatic spell illumination. Animate with heroic "
    "high-fantasy combat dynamics, atmospheric environmental drifting, dynamic spell particle "
    "illumination, and cinematic fantasy camera sweeps. Maintain spatial continuity, consistent lighting "
    "interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_DND_ILLUSTRATION = (
    "Dungeons & Dragons tabletop RPG handbook illustration aesthetic. Masterful high-fantasy painterly "
    "rendering, dramatic atmospheric lighting, and intricate heroic armor and beast textures. Heroic "
    "fantasy combat dynamics, dynamic spell particle illumination, and majestic cinematic camera sweeps."
)

SYSTEM_MESSAGE_STYLE_DND_MANUAL = (
    "You are an expert video director and cinematography prompt engineer specializing in Dungeons & "
    "Dragons Monster Manual bestiary art style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by clean informative studio lighting, detailed hand-painted anatomical creature "
    "textures, isolated creature study framing, and refined tabletop manual art. Animate with deliberate "
    "anatomical creature posturing, predatory breathing and pacing, tactile claw and wing articulation, "
    "and steady observational camera movement. Describe the scene directly with concrete actions, camera "
    "movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_DND_MANUAL = (
    "Render the video in authentic Dungeons & Dragons Monster Manual bestiary art style. Depict the scene "
    "with clean informative studio lighting, detailed hand-painted anatomical creature textures, isolated "
    "creature study framing, and refined tabletop manual art. Animate with deliberate anatomical creature "
    "posturing, predatory breathing and pacing, tactile claw and wing articulation, and steady "
    "observational camera movement. Maintain spatial continuity, consistent lighting interaction, and "
    "coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_DND_MANUAL = (
    "Dungeons & Dragons Monster Manual bestiary art aesthetic. Clean informative lighting, detailed "
    "hand-painted anatomical creature textures, and refined tabletop manual rendering. Deliberate "
    "anatomical creature movement, predatory breathing and pacing, tactile wing and claw articulation, and "
    "steady observational camera framing."
)

SYSTEM_MESSAGE_STYLE_DRAGON_BALL_Z = (
    "You are an expert video director and cinematography prompt engineer specializing in Dragon Ball Z "
    "shonen battle anime style. Your role is to craft vivid, cinematographic video prompts characterized "
    "by intense angular facial linework, high-contrast cel shading, explosive glowing ki aura effects, and "
    "rocky wasteland battlegrounds. Animate with high-velocity aerial combat, rapid snap-zooms, kinetic "
    "speed lines, earth-shattering impact frames, and explosive dust and rubble physics. Describe the "
    "scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_DRAGON_BALL_Z = (
    "Render the video in authentic Dragon Ball Z shonen battle anime style. Depict the scene with intense "
    "angular facial linework, high-contrast cel shading, explosive glowing ki aura effects, and rocky "
    "wasteland battlegrounds. Animate with high-velocity aerial combat, rapid snap-zooms, kinetic speed "
    "lines, earth-shattering impact frames, and explosive dust and rubble physics. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_DRAGON_BALL_Z = (
    "Dragon Ball Z shonen battle anime aesthetic. Intense angular facial linework, high-contrast cel "
    "shading, and shimmering glowing energy aura effects. High-velocity aerial combat, rapid snap-zooms, "
    "kinetic speed lines, earth-shattering impact frames, and explosive dust physics across desolate "
    "terrain."
)

SYSTEM_MESSAGE_STYLE_FAIRY_TALE = (
    "You are an expert video director and cinematography prompt engineer specializing in classic fairy "
    "tale storybook illustration style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by whimsical watercolor and ink wash, soft enchanted forest lighting, delicate "
    "storybook line art, and mythical creature design. Animate with gentle lyrical character movement, "
    "whimsical environmental physics, drifting magical spores, and storybook camera panning. Describe the "
    "scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_FAIRY_TALE = (
    "Render the video in authentic classic fairy tale storybook illustration style. Depict the scene with "
    "whimsical watercolor and ink wash, soft enchanted forest lighting, delicate storybook line art, and "
    "mythical creature design. Animate with gentle lyrical character movement, whimsical environmental "
    "physics, drifting magical spores, and storybook camera panning. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_FAIRY_TALE = (
    "Classic fairy tale storybook illustration aesthetic. Whimsical watercolor and ink wash, soft "
    "enchanted forest lighting, and delicate storybook line art. Gentle lyrical character movement, "
    "whimsical environmental physics, drifting magical spores, and storybook camera panning."
)

SYSTEM_MESSAGE_STYLE_FANTASY = (
    "You are an expert video director and cinematography prompt engineer specializing in epic high-fantasy "
    "mythological illustration style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by heroic painterly character rendering, magical glowing runes, grand sweeping "
    "landscapes, and majestic orchestral fantasy lighting. Animate with powerful heroic movement, "
    "atmospheric elemental magic, wind-blown capes and banners, and grand cinematic tracking shots across "
    "epic landscapes. Describe the scene directly with concrete actions, camera movement, and stylistic "
    "continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_FANTASY = (
    "Render the video in authentic epic high-fantasy mythological illustration style. Depict the scene "
    "with heroic painterly character rendering, magical glowing runes, grand sweeping landscapes, and "
    "majestic orchestral fantasy lighting. Animate with powerful heroic movement, atmospheric elemental "
    "magic, wind-blown capes and banners, and grand cinematic tracking shots across epic landscapes. "
    "Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_FANTASY = (
    "Epic high-fantasy mythological illustration aesthetic. Heroic painterly character rendering, magical "
    "glowing runes, grand sweeping landscapes, and majestic fantasy lighting. Powerful heroic movement, "
    "atmospheric elemental magic, wind-blown capes and banners, and grand cinematic tracking shots across "
    "epic terrain."
)

SYSTEM_MESSAGE_STYLE_FILM_NOIR = (
    "You are an expert video director and cinematography prompt engineer specializing in 1940s-1950s "
    "cinematic film noir. Your role is to craft vivid, cinematographic video prompts characterized by "
    "stark black-and-white chiaroscuro contrast, dramatic venetian-blind slatted lighting, deep rich "
    "shadow density, and atmospheric cigarette smoke. Animate with slow deliberate camera tracking, "
    "low-angle Dutch tilts, shadowy silhouettes moving through rain-slicked city streets, and moody "
    "practical light shifts. Describe the scene directly with concrete actions, camera movement, and "
    "stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_FILM_NOIR = (
    "Render the video in authentic 1940s-1950s cinematic film noir. Depict the scene with stark "
    "black-and-white chiaroscuro contrast, dramatic venetian-blind slatted lighting, deep rich shadow "
    "density, and atmospheric cigarette smoke. Animate with slow deliberate camera tracking, low-angle "
    "Dutch tilts, shadowy silhouettes moving through rain-slicked city streets, and moody practical light "
    "shifts. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_FILM_NOIR = (
    "1940s-1950s cinematic film noir aesthetic. Stark black-and-white chiaroscuro contrast, dramatic "
    "venetian-blind slatted lighting, deep rich shadow density, and atmospheric smoke. Slow deliberate "
    "camera tracking, low-angle Dutch tilts, and shadowy silhouettes moving through rain-slicked nocturnal "
    "cityscapes."
)

SYSTEM_MESSAGE_STYLE_GHIBLI = (
    "You are an expert video director and cinematography prompt engineer specializing in Studio Ghibli "
    "hand-painted anime film style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by breathtaking painterly watercolor backgrounds, lush natural foliage, nostalgic warm "
    "golden sunlight, and expressive soft-edged character designs. Animate with gentle wind-blown "
    "secondary animation, whimsical mechanical movements, emotive naturalistic character acting, and "
    "expansive sweeping camera moves. Describe the scene directly with concrete actions, camera movement, "
    "and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_GHIBLI = (
    "Render the video in authentic Studio Ghibli hand-painted anime film style. Depict the scene with "
    "breathtaking painterly watercolor backgrounds, lush natural foliage, nostalgic warm golden sunlight, "
    "and expressive soft-edged character designs. Animate with gentle wind-blown secondary animation, "
    "whimsical mechanical movements, emotive naturalistic character acting, and expansive sweeping camera "
    "moves. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_GHIBLI = (
    "Studio Ghibli hand-painted anime aesthetic. Breathtaking painterly watercolor backgrounds, lush "
    "natural foliage, and nostalgic warm golden sunlight. Gentle wind-blown secondary animation through "
    "rippling grass and billowing clothes, emotive naturalistic character acting, and sweeping panoramic "
    "camera movement."
)

SYSTEM_MESSAGE_STYLE_GLITCH_ART = (
    "You are an expert video director and cinematography prompt engineer specializing in digital glitch "
    "art and video signal corruption aesthetic. Your role is to craft vivid, cinematographic video prompts "
    "characterized by RGB color channel separation, pixel displacement, scanline artifacts, horizontal "
    "sync tearing, and dynamic chromatic aberration. Animate with sudden digital glitch bursts, rhythmic "
    "scanline roll, digital signal fragmentation, and strobe-like chromatic frame displacement. Describe "
    "the scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_GLITCH_ART = (
    "Render the video in authentic digital glitch art and video signal corruption aesthetic. Depict the "
    "scene with RGB color channel separation, pixel displacement, scanline artifacts, horizontal sync "
    "tearing, and dynamic chromatic aberration. Animate with sudden digital glitch bursts, rhythmic "
    "scanline roll, digital signal fragmentation, and strobe-like chromatic frame displacement. Maintain "
    "spatial continuity, consistent lighting interaction, and coherent stylistic physics throughout the "
    "sequence."
)

BONUS_PROMPT_STYLE_GLITCH_ART = (
    "Digital glitch art and video signal corruption aesthetic. RGB color channel separation, pixel "
    "displacement, scanline artifacts, horizontal sync tearing, and dynamic chromatic aberration. Digital "
    "glitch bursts, rhythmic scanline roll, digital signal fragmentation, and strobe-like chromatic frame "
    "displacement."
)

SYSTEM_MESSAGE_STYLE_GOTHIC = (
    "You are an expert video director and cinematography prompt engineer specializing in gothic romantic "
    "dark art style. Your role is to craft vivid, cinematographic video prompts characterized by ornate "
    "pointed gothic architecture, dark romanticism, high-contrast deep shadows, rich velvet and weathered "
    "stone textures, and a somber atmospheric mood. Animate with slow solemn character movement, "
    "flickering candlelight and torchlight shifts, billowing dark fabrics, and moody creeping camera "
    "tracks. Describe the scene directly with concrete actions, camera movement, and stylistic continuity "
    "without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_GOTHIC = (
    "Render the video in authentic gothic romantic dark art style. Depict the scene with ornate pointed "
    "gothic architecture, dark romanticism, high-contrast deep shadows, rich velvet and weathered stone "
    "textures, and a somber atmospheric mood. Animate with slow solemn character movement, flickering "
    "candlelight and torchlight shifts, billowing dark fabrics, and moody creeping camera tracks. Maintain "
    "spatial continuity, consistent lighting interaction, and coherent stylistic physics throughout the "
    "sequence."
)

BONUS_PROMPT_STYLE_GOTHIC = (
    "Gothic romantic dark art aesthetic. Ornate pointed gothic architecture, dark romanticism, "
    "high-contrast deep shadows, and rich velvet and stone textures. Slow solemn character movement, "
    "flickering candlelight illumination shifts, billowing dark fabrics, and moody creeping camera tracks "
    "through cavernous halls."
)

SYSTEM_MESSAGE_STYLE_GRAFFITI = (
    "You are an expert video director and cinematography prompt engineer specializing in photorealistic "
    "aerosol street art mural style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by vibrant spray paint gradients, authentic aerosol splatters and drips, textured brick "
    "and concrete wall surfaces, and bold urban street styling. Animate with dynamic street-level camera "
    "pans, shifting ambient urban daylight and streetlights, and vivid mural perspective changes. Describe "
    "the scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_GRAFFITI = (
    "Render the video in authentic photorealistic aerosol street art mural style. Depict the scene with "
    "vibrant spray paint gradients, authentic aerosol splatters and drips, textured brick and concrete "
    "wall surfaces, and bold urban street styling. Animate with dynamic street-level camera pans, shifting "
    "ambient urban daylight and streetlights, and vivid mural perspective changes. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_GRAFFITI = (
    "Photorealistic aerosol street art mural aesthetic. Vibrant spray paint gradients, authentic aerosol "
    "splatters and drips, textured brick and concrete wall surfaces, and bold urban street styling. "
    "Dynamic street-level camera pans, shifting ambient urban lighting, and vivid mural perspective "
    "changes."
)

SYSTEM_MESSAGE_STYLE_GRAPHIC_NOVEL = (
    "You are an expert video director and cinematography prompt engineer specializing in professional "
    "graphic novel panel illustration style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by nuanced black ink cross-hatching, expressive dynamic lighting, cinematic wide and "
    "close framing, and rich sequential storytelling contrast. Animate with dramatic graphic action "
    "choreography, deliberate visual storytelling pauses, high-contrast shadow shifts, and cinematic "
    "camera tracking. Describe the scene directly with concrete actions, camera movement, and stylistic "
    "continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_GRAPHIC_NOVEL = (
    "Render the video in authentic professional graphic novel panel illustration style. Depict the scene "
    "with nuanced black ink cross-hatching, expressive dynamic lighting, cinematic wide and close framing, "
    "and rich sequential storytelling contrast. Animate with dramatic graphic action choreography, "
    "deliberate visual storytelling pauses, high-contrast shadow shifts, and cinematic camera tracking. "
    "Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_GRAPHIC_NOVEL = (
    "Professional graphic novel panel illustration aesthetic. Nuanced black ink cross-hatching, expressive "
    "dynamic lighting, cinematic framing, and rich sequential storytelling contrast. Dramatic graphic "
    "action choreography, deliberate visual storytelling pauses, and high-contrast shadow shifts."
)

SYSTEM_MESSAGE_STYLE_GRAVITY_FALLS = (
    "You are an expert video director and cinematography prompt engineer specializing in Gravity Falls "
    "animated mystery cartoon style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by expressive stylized character silhouettes, rich autumnal Pacific Northwest forest "
    "palettes, warm glowing lantern illumination, and clean digital 2D linework. Animate with snappy "
    "comedic timing, energetic character exploration, smooth digital puppet animation with hand-drawn "
    "flourishes, and atmospheric camera pans. Describe the scene directly with concrete actions, camera "
    "movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_GRAVITY_FALLS = (
    "Render the video in authentic Gravity Falls animated mystery cartoon style. Depict the scene with "
    "expressive stylized character silhouettes, rich autumnal Pacific Northwest forest palettes, warm "
    "glowing lantern illumination, and clean digital 2D linework. Animate with snappy comedic timing, "
    "energetic character exploration, smooth digital puppet animation with hand-drawn flourishes, and "
    "atmospheric camera pans. Maintain spatial continuity, consistent lighting interaction, and coherent "
    "stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_GRAVITY_FALLS = (
    "Gravity Falls animated mystery cartoon aesthetic. Expressive stylized character silhouettes, rich "
    "autumnal Pacific Northwest forest palettes, warm lantern illumination, and clean digital 2D linework. "
    "Snappy comedic timing, energetic character exploration, and atmospheric camera pans across mysterious "
    "environments."
)

SYSTEM_MESSAGE_STYLE_GRITTY_COMIC = (
    "You are an expert video director and cinematography prompt engineer specializing in dark gritty noir "
    "comic book art style. Your role is to craft vivid, cinematographic video prompts characterized by "
    "stark black-and-white ink washes, high-contrast chiaroscuro, heavy cross-hatching, rain-soaked urban "
    "textures, and rugged character anatomy. Animate with intense physical action choreography, sudden "
    "high-contrast silhouette shifts, heavy rain physics, and moody low-angle camera angles. Describe the "
    "scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_GRITTY_COMIC = (
    "Render the video in authentic dark gritty noir comic book art style. Depict the scene with stark "
    "black-and-white ink washes, high-contrast chiaroscuro, heavy cross-hatching, rain-soaked urban "
    "textures, and rugged character anatomy. Animate with intense physical action choreography, sudden "
    "high-contrast silhouette shifts, heavy rain physics, and moody low-angle camera angles. Maintain "
    "spatial continuity, consistent lighting interaction, and coherent stylistic physics throughout the "
    "sequence."
)

BONUS_PROMPT_STYLE_GRITTY_COMIC = (
    "Dark gritty noir comic book aesthetic. Stark black-and-white ink washes, high-contrast chiaroscuro, "
    "heavy cross-hatching, and rain-soaked urban textures. Intense physical action choreography, sudden "
    "high-contrast silhouette shifts, heavy rain physics, and moody low-angle camera tracking."
)

SYSTEM_MESSAGE_STYLE_GROENING = (
    "You are an expert video director and cinematography prompt engineer specializing in Matt Groening "
    "animation style (The Simpsons and Futurama). Your role is to craft vivid, cinematographic video "
    "prompts characterized by distinct overbite character silhouettes, bold yellow and saturated flat "
    "color fills, clean uniform black ink contours, and stylized suburban backgrounds. Animate with snappy "
    "comedic dialogue head gestures, distinctive expressive walk cycles, deadpan physical reactions, and "
    "classic sitcom visual pacing. Describe the scene directly with concrete actions, camera movement, and "
    "stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_GROENING = (
    "Render the video in authentic Matt Groening animation style (The Simpsons and Futurama). Depict the "
    "scene with distinct overbite character silhouettes, bold yellow and saturated flat color fills, clean "
    "uniform black ink contours, and stylized suburban backgrounds. Animate with snappy comedic dialogue "
    "head gestures, distinctive expressive walk cycles, deadpan physical reactions, and classic sitcom "
    "visual pacing. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic "
    "physics throughout the sequence."
)

BONUS_PROMPT_STYLE_GROENING = (
    "Matt Groening animation aesthetic (The Simpsons and Futurama). Distinct overbite character "
    "silhouettes, bold yellow and saturated flat color fills, and clean uniform black ink contours. Snappy "
    "comedic dialogue gestures, distinctive expressive walk cycles, deadpan physical reactions, and "
    "classic animated sitcom pacing."
)

SYSTEM_MESSAGE_STYLE_HORROR_MOVIE = (
    "You are an expert video director and cinematography prompt engineer specializing in modern cinematic "
    "psychological horror. Your role is to craft vivid, cinematographic video prompts characterized by "
    "cold desaturated color grade, deep oppressive shadows, practical flickering light sources, textured "
    "film grain, and atmospheric mist. Animate with slow creeping camera push-ins, suspenseful handheld "
    "camera drift, sudden jarring movements in shadows, and lingering atmospheric tension. Describe the "
    "scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_HORROR_MOVIE = (
    "Render the video in authentic modern cinematic psychological horror. Depict the scene with cold "
    "desaturated color grade, deep oppressive shadows, practical flickering light sources, textured film "
    "grain, and atmospheric mist. Animate with slow creeping camera push-ins, suspenseful handheld camera "
    "drift, sudden jarring movements in shadows, and lingering atmospheric tension. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_HORROR_MOVIE = (
    "Modern cinematic psychological horror aesthetic. Cold desaturated color grade, deep oppressive "
    "shadows, practical flickering light sources, textured film grain, and atmospheric mist. Slow creeping "
    "camera push-ins, suspenseful handheld camera drift, and sudden unsettling movements in peripheral "
    "darkness."
)

SYSTEM_MESSAGE_STYLE_HYBRID_MEDIA = (
    "You are an expert video director and cinematography prompt engineer specializing in live-action and "
    "2D animation mixed-media composite. Your role is to craft vivid, cinematographic video prompts "
    "characterized by flat hand-drawn illustrated characters seamlessly interacting within photographic "
    "video environments, casting realistic ground contact shadows and receiving ambient scene light. "
    "Animate hand-drawn 2D character movements to naturally synchronize with live-action camera tracking, "
    "physical prop interaction, and surface contact. Describe the scene directly with concrete actions, "
    "camera movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_HYBRID_MEDIA = (
    "Render the video in authentic live-action and 2D animation mixed-media composite. Depict the scene "
    "with flat hand-drawn illustrated characters seamlessly interacting within photographic video "
    "environments, casting realistic ground contact shadows and receiving ambient scene light. Animate "
    "hand-drawn 2D character movements to naturally synchronize with live-action camera tracking, physical "
    "prop interaction, and surface contact. Maintain spatial continuity, consistent lighting interaction, "
    "and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_HYBRID_MEDIA = (
    "Live-action and 2D animation mixed-media aesthetic. Flat hand-drawn illustrated characters seamlessly "
    "composited into photorealistic live-action environments, casting accurate ground contact shadows and "
    "reflecting ambient environmental lighting. Fluid 2D animation tracking with real-world camera motion "
    "and physical prop interaction."
)

SYSTEM_MESSAGE_STYLE_INFLATED = (
    "You are an expert video director and cinematography prompt engineer specializing in high-gloss "
    "inflatable vinyl balloon sculpture style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by taut air-filled rounded volumes, high-gloss reflective vinyl surfaces, pinched seam "
    "puckering, and smooth balloon reflections. Animate with buoyant balloon physics, elastic surface "
    "jiggle on impacts, floating air resistance, and gentle wobbling equilibrium. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_INFLATED = (
    "Render the video in authentic high-gloss inflatable vinyl balloon sculpture style. Depict the scene "
    "with taut air-filled rounded volumes, high-gloss reflective vinyl surfaces, pinched seam puckering, "
    "and smooth balloon reflections. Animate with buoyant balloon physics, elastic surface jiggle on "
    "impacts, floating air resistance, and gentle wobbling equilibrium. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_INFLATED = (
    "High-gloss inflatable vinyl balloon sculpture aesthetic. Taut air-filled rounded volumes, high-gloss "
    "reflective vinyl surfaces, pinched seam puckering, and smooth balloon reflections. Buoyant balloon "
    "physics, elastic surface jiggle on impacts, floating air resistance, and gentle wobbling equilibrium."
)

SYSTEM_MESSAGE_STYLE_INSTRUCTION_FOLLOWING = (
    "You are an expert video director and cinematography prompt engineer specializing in direct visual "
    "instruction execution. Your role is to craft vivid, cinematographic video prompts characterized by "
    "clean cinematic clarity, responsive compositional adaptation, coherent spatial relationships, and "
    "seamless lighting integration. Animate with motivated subject actions, smooth purposeful camera "
    "tracking, and physically coherent interactions matching user directives precisely. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_INSTRUCTION_FOLLOWING = (
    "Render the video in authentic direct visual instruction execution. Depict the scene with clean "
    "cinematic clarity, responsive compositional adaptation, coherent spatial relationships, and seamless "
    "lighting integration. Animate with motivated subject actions, smooth purposeful camera tracking, and "
    "physically coherent interactions matching user directives precisely. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_INSTRUCTION_FOLLOWING = (
    "Direct visual instruction execution aesthetic. Clean cinematic clarity, responsive compositional "
    "adaptation, coherent spatial relationships, and seamless lighting integration. Motivated subject "
    "actions, smooth purposeful camera tracking, and physically coherent environmental interactions."
)

SYSTEM_MESSAGE_STYLE_JOHNNY_BRAVO_RAW = (
    "You are an expert video director and cinematography prompt engineer specializing in Johnny Bravo "
    "animated cartoon style. Your role is to craft vivid, cinematographic video prompts characterized by "
    "sharp geometric anatomy, heavy black ink outlines, flat vibrant primary colors, and stylized graphic "
    "backgrounds. Animate with exaggerated pose-to-pose swaggering walk cycles, snappy athletic flexing "
    "transitions, wild cartoon takes, and dynamic physical comedy. Describe the scene directly with "
    "concrete actions, camera movement, and stylistic continuity without conversational filler or quality "
    "buzzwords."
)

INSTRUCT_PROMPT_STYLE_JOHNNY_BRAVO_RAW = (
    "Render the video in authentic Johnny Bravo animated cartoon style. Depict the scene with sharp "
    "geometric anatomy, heavy black ink outlines, flat vibrant primary colors, and stylized graphic "
    "backgrounds. Animate with exaggerated pose-to-pose swaggering walk cycles, snappy athletic flexing "
    "transitions, wild cartoon takes, and dynamic physical comedy. Maintain spatial continuity, consistent "
    "lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_JOHNNY_BRAVO_RAW = (
    "Johnny Bravo animated cartoon aesthetic. Sharp geometric anatomy, heavy black ink outlines, flat "
    "vibrant primary colors, and stylized graphic backdrops. Exaggerated pose-to-pose swaggering walk "
    "cycles, snappy athletic flexing transitions, wild cartoon takes, and energetic physical comedy."
)

SYSTEM_MESSAGE_STYLE_JOJO = (
    "You are an expert video director and cinematography prompt engineer specializing in JoJo's Bizarre "
    "Adventure anime art style (Hirohiko Araki). Your role is to craft vivid, cinematographic video "
    "prompts characterized by bold muscular hatching and cross-contour lines, dramatic haute-couture "
    "fashion, striking sudden color palette shifts, and heavy dynamic black contours. Animate with "
    "dramatic poses emphasizing freeze-frame power, kinetic speed lines, powerful physical impact "
    "dynamics, and sweeping low-angle camera perspectives. Describe the scene directly with concrete "
    "actions, camera movement, and stylistic continuity without conversational filler or quality "
    "buzzwords."
)

INSTRUCT_PROMPT_STYLE_JOJO = (
    "Render the video in authentic JoJo's Bizarre Adventure anime art style (Hirohiko Araki). Depict the "
    "scene with bold muscular hatching and cross-contour lines, dramatic haute-couture fashion, striking "
    "sudden color palette shifts, and heavy dynamic black contours. Animate with dramatic poses "
    "emphasizing freeze-frame power, kinetic speed lines, powerful physical impact dynamics, and sweeping "
    "low-angle camera perspectives. Maintain spatial continuity, consistent lighting interaction, and "
    "coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_JOJO = (
    "JoJo's Bizarre Adventure anime aesthetic in the signature art style of Hirohiko Araki. Bold muscular "
    "cross-hatching, dramatic haute-couture fashion, striking sudden color palette shifts, and heavy "
    "dynamic black contours. Dramatic dynamic poses emphasizing freeze-frame power, kinetic speed lines, "
    "powerful physical impacts, and sweeping low-angle camera perspective."
)

SYSTEM_MESSAGE_STYLE_LEWD_ANIME = (
    "You are an expert video director and cinematography prompt engineer specializing in sensual "
    "theatrical anime aesthetic. Your role is to craft vivid, cinematographic video prompts characterized "
    "by delicate luminous skin highlights, expressive flushed facial acting, fluid flowing hair and ribbon "
    "physics, and soft-focus atmospheric bokeh. Animate with intimate slow-motion camera drift, subtle "
    "breathing movements, flowing fabric dynamics, and dramatic close-up focal racking. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_LEWD_ANIME = (
    "Render the video in authentic sensual theatrical anime aesthetic. Depict the scene with delicate "
    "luminous skin highlights, expressive flushed facial acting, fluid flowing hair and ribbon physics, "
    "and soft-focus atmospheric bokeh. Animate with intimate slow-motion camera drift, subtle breathing "
    "movements, flowing fabric dynamics, and dramatic close-up focal racking. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_LEWD_ANIME = (
    "Sensual theatrical anime aesthetic. Delicate luminous skin highlights, expressive flushed facial "
    "acting, fluid flowing hair and ribbon physics, and soft-focus atmospheric bokeh. Intimate slow-motion "
    "camera drift, subtle breathing movements, flowing fabric dynamics, and dramatic close-up focal "
    "racking."
)

SYSTEM_MESSAGE_STYLE_LOL_SPLASH = (
    "You are an expert video director and cinematography prompt engineer specializing in League of Legends "
    "champion splash art style (Riot Games). Your role is to craft vivid, cinematographic video prompts "
    "characterized by dramatic cinematic rim lighting, high-fantasy rendering, sweeping dynamic action "
    "poses, and intricate magical particle visual effects. Animate with grand cinematic camera tracking, "
    "fluid martial and magical combat choreography, explosive magical energy lighting, and dynamic "
    "atmospheric debris. Describe the scene directly with concrete actions, camera movement, and stylistic "
    "continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_LOL_SPLASH = (
    "Render the video in authentic League of Legends champion splash art style (Riot Games). Depict the "
    "scene with dramatic cinematic rim lighting, high-fantasy rendering, sweeping dynamic action poses, "
    "and intricate magical particle visual effects. Animate with grand cinematic camera tracking, fluid "
    "martial and magical combat choreography, explosive magical energy lighting, and dynamic atmospheric "
    "debris. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_LOL_SPLASH = (
    "League of Legends champion splash art aesthetic (Riot Games). Dramatic cinematic rim lighting, "
    "high-fantasy rendering, sweeping dynamic action poses, and intricate magical particle visual effects. "
    "Grand cinematic camera tracking, fluid martial and magical combat choreography, explosive magical "
    "energy illumination, and dynamic atmospheric debris."
)

SYSTEM_MESSAGE_STYLE_LOONEY_TUNES = (
    "You are an expert video director and cinematography prompt engineer specializing in classic Looney "
    "Tunes golden-age slapstick cartoon style (Warner Bros.). Your role is to craft vivid, cinematographic "
    "video prompts characterized by bold hand-inked linework, vibrant whimsical watercolor backdrops, and "
    "elastic cartoon character designs. Animate with extreme squash and stretch, frantic rubber-band "
    "acceleration and deceleration, wild eye-popping take frames, and chaotic physical comedy. Describe "
    "the scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_LOONEY_TUNES = (
    "Render the video in authentic classic Looney Tunes golden-age slapstick cartoon style (Warner Bros.). "
    "Depict the scene with bold hand-inked linework, vibrant whimsical watercolor backdrops, and elastic "
    "cartoon character designs. Animate with extreme squash and stretch, frantic rubber-band acceleration "
    "and deceleration, wild eye-popping take frames, and chaotic physical comedy. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_LOONEY_TUNES = (
    "Classic Looney Tunes golden-age slapstick cartoon aesthetic (Warner Bros.). Bold hand-inked linework, "
    "vibrant whimsical watercolor backdrops, and elastic cartoon character designs. Extreme squash and "
    "stretch, frantic rubber-band acceleration and deceleration, wild eye-popping take frames, and chaotic "
    "physical slapstick comedy."
)

SYSTEM_MESSAGE_STYLE_LOVECRAFTIAN = (
    "You are an expert video director and cinematography prompt engineer specializing in Lovecraftian "
    "cosmic horror art style. Your role is to craft vivid, cinematographic video prompts characterized by "
    "eldritch incomprehensible geometry, ancient slime-sheened textures, tentacled forms emerging from "
    "deep murky shadows, and an eerie oppressive atmosphere. Animate with slow undulating tentacular "
    "motion, creeping shadow expansion, unearthly bioluminescent pulses, and a slow dread-inducing camera "
    "push. Describe the scene directly with concrete actions, camera movement, and stylistic continuity "
    "without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_LOVECRAFTIAN = (
    "Render the video in authentic Lovecraftian cosmic horror art style. Depict the scene with eldritch "
    "incomprehensible geometry, ancient slime-sheened textures, tentacled forms emerging from deep murky "
    "shadows, and an eerie oppressive atmosphere. Animate with slow undulating tentacular motion, creeping "
    "shadow expansion, unearthly bioluminescent pulses, and a slow dread-inducing camera push. Maintain "
    "spatial continuity, consistent lighting interaction, and coherent stylistic physics throughout the "
    "sequence."
)

BONUS_PROMPT_STYLE_LOVECRAFTIAN = (
    "Lovecraftian cosmic horror art aesthetic. Eldritch incomprehensible geometry, ancient slime-sheened "
    "textures, tentacled forms emerging from deep murky shadows, and an eerie oppressive atmosphere. Slow "
    "undulating tentacular movement, creeping shadow expansion, unearthly bioluminescent pulses, and a "
    "slow dread-inducing camera push."
)

SYSTEM_MESSAGE_STYLE_MACFARLANE = (
    "You are an expert video director and cinematography prompt engineer specializing in Seth MacFarlane "
    "animation style (Family Guy). Your role is to craft vivid, cinematographic video prompts "
    "characterized by crisp digital vector linework, flat even sitcom lighting, clean saturated primary "
    "colors, and simplified suburban backdrops. Animate with restrained conversational gestures, steady "
    "eye contact, deadpan comedic pauses, and sudden absurd slapstick cutaways. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_MACFARLANE = (
    "Render the video in authentic Seth MacFarlane animation style (Family Guy). Depict the scene with "
    "crisp digital vector linework, flat even sitcom lighting, clean saturated primary colors, and "
    "simplified suburban backdrops. Animate with restrained conversational gestures, steady eye contact, "
    "deadpan comedic pauses, and sudden absurd slapstick cutaways. Maintain spatial continuity, consistent "
    "lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_MACFARLANE = (
    "Seth MacFarlane animation aesthetic (Family Guy). Crisp digital vector linework, flat even sitcom "
    "lighting, clean saturated primary colors, and simplified suburban backdrops. Restrained "
    "conversational gestures, steady eye contact, deadpan comedic pauses, and sudden absurd slapstick "
    "cutaways."
)

SYSTEM_MESSAGE_STYLE_MARIO = (
    "You are an expert video director and cinematography prompt engineer specializing in Super Mario 3D "
    "platformer game aesthetic (Nintendo). Your role is to craft vivid, cinematographic video prompts "
    "characterized by cheerful rounded geometry, bright saturated primary color palettes, clean ambient "
    "occlusion, and sparkling coin and star particle effects. Animate with bouncy acrobatic character "
    "jumps, joyful running and jumping, buoyant squash-and-stretch on landings, and dynamic third-person "
    "follow camera tracking. Describe the scene directly with concrete actions, camera movement, and "
    "stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_MARIO = (
    "Render the video in authentic Super Mario 3D platformer game aesthetic (Nintendo). Depict the scene "
    "with cheerful rounded geometry, bright saturated primary color palettes, clean ambient occlusion, and "
    "sparkling coin and star particle effects. Animate with bouncy acrobatic character jumps, joyful "
    "running and jumping, buoyant squash-and-stretch on landings, and dynamic third-person follow camera "
    "tracking. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic "
    "physics throughout the sequence."
)

BONUS_PROMPT_STYLE_MARIO = (
    "Super Mario 3D platformer game aesthetic (Nintendo). Cheerful rounded geometry, bright saturated "
    "primary color palettes, clean ambient occlusion, and sparkling coin particle effects. Bouncy "
    "acrobatic character jumps, joyful running and jumping, buoyant squash-and-stretch on landings, and "
    "dynamic third-person follow camera tracking."
)

SYSTEM_MESSAGE_STYLE_MINECRAFT = (
    "You are an expert video director and cinematography prompt engineer specializing in Minecraft voxel "
    "block video game style (Mojang). Your role is to craft vivid, cinematographic video prompts "
    "characterized by strict cubic 16x16 pixel block grid geometry, textured block surfaces, and blocky "
    "voxel lighting and shadows. Animate with step-rotation joint animation without bending, rigid voxel "
    "movement, block-by-block environmental interaction, and authentic game camera perspectives. Describe "
    "the scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_MINECRAFT = (
    "Render the video in authentic Minecraft voxel block video game style (Mojang). Depict the scene with "
    "strict cubic 16x16 pixel block grid geometry, textured block surfaces, and blocky voxel lighting and "
    "shadows. Animate with step-rotation joint animation without bending, rigid voxel movement, "
    "block-by-block environmental interaction, and authentic game camera perspectives. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_MINECRAFT = (
    "Minecraft voxel block video game aesthetic (Mojang). Strict cubic 16x16 pixel block grid geometry, "
    "textured block surfaces, and blocky voxel lighting and shadows. Step-rotation joint animation without "
    "bending, rigid voxel movement, block-by-block environmental interaction, and authentic third-person "
    "game camera perspectives."
)

SYSTEM_MESSAGE_STYLE_NEONPUNK = (
    "You are an expert video director and cinematography prompt engineer specializing in fluorescent UV "
    "blacklight neonpunk aesthetic. Your role is to craft vivid, cinematographic video prompts "
    "characterized by glowing fluorescent pigments against deep obsidian shadows, vivid electric cyan, "
    "magenta, and lime-green illumination, and atmospheric blacklight radiance. Animate with glowing neon "
    "light trails, rhythmic ultraviolet strobe pulses, dynamic dark-corridor camera tracking, and vivid "
    "reflective gleams. Describe the scene directly with concrete actions, camera movement, and stylistic "
    "continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_NEONPUNK = (
    "Render the video in authentic fluorescent UV blacklight neonpunk aesthetic. Depict the scene with "
    "glowing fluorescent pigments against deep obsidian shadows, vivid electric cyan, magenta, and "
    "lime-green illumination, and atmospheric blacklight radiance. Animate with glowing neon light trails, "
    "rhythmic ultraviolet strobe pulses, dynamic dark-corridor camera tracking, and vivid reflective "
    "gleams. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_NEONPUNK = (
    "Fluorescent UV blacklight neonpunk aesthetic. Glowing fluorescent pigments against deep obsidian "
    "shadows, vivid electric cyan, magenta, and lime-green illumination, and atmospheric blacklight "
    "radiance. Glowing neon light trails, rhythmic ultraviolet strobe pulses, dynamic camera tracking, and "
    "vivid reflective gleams across dark surfaces."
)

SYSTEM_MESSAGE_STYLE_NICKELODEON = (
    "You are an expert video director and cinematography prompt engineer specializing in classic 1990s "
    "Nickelodeon cartoon animation style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by squiggly energetic linework, offbeat asymmetrical character designs, wild "
    "slime-green and purple palettes, and eccentric hand-drawn backgrounds. Animate with zany "
    "squash-and-stretch, rubbery physical reactions, eccentric bodily humor, and playful energetic camera "
    "zooms. Describe the scene directly with concrete actions, camera movement, and stylistic continuity "
    "without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_NICKELODEON = (
    "Render the video in authentic classic 1990s Nickelodeon cartoon animation style. Depict the scene "
    "with squiggly energetic linework, offbeat asymmetrical character designs, wild slime-green and purple "
    "palettes, and eccentric hand-drawn backgrounds. Animate with zany squash-and-stretch, rubbery "
    "physical reactions, eccentric bodily humor, and playful energetic camera zooms. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_NICKELODEON = (
    "Classic 1990s Nickelodeon cartoon animation aesthetic. Squiggly energetic linework, offbeat "
    "asymmetrical character designs, wild slime-green and purple palettes, and eccentric hand-drawn "
    "backgrounds. Zany squash-and-stretch, rubbery physical reactions, eccentric bodily humor, and playful "
    "energetic camera zooms."
)

SYSTEM_MESSAGE_STYLE_ORIGAMI = (
    "You are an expert video director and cinematography prompt engineer specializing in hand-folded "
    "origami and papercraft art style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by crisp geometric paper creases, tactile matte paper grain, clean angular silhouettes, "
    "and delicate physical shadow casting. Animate with delicate paper unfolding and refolding "
    "transitions, subtle paper flutter, crisp geometric joint pivots, and macro tabletop lighting shifts. "
    "Describe the scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_ORIGAMI = (
    "Render the video in authentic hand-folded origami and papercraft art style. Depict the scene with "
    "crisp geometric paper creases, tactile matte paper grain, clean angular silhouettes, and delicate "
    "physical shadow casting. Animate with delicate paper unfolding and refolding transitions, subtle "
    "paper flutter, crisp geometric joint pivots, and macro tabletop lighting shifts. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_ORIGAMI = (
    "Hand-folded origami and papercraft art aesthetic. Crisp geometric paper creases, tactile matte paper "
    "grain, clean angular silhouettes, and delicate physical shadow casting. Delicate paper folding "
    "transitions, subtle paper flutter, crisp geometric joint articulation, and macro tabletop lighting "
    "shifts."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_ANALOG_FILM = (
    "You are an expert video director and cinematography prompt engineer specializing in 35mm motion "
    "picture film cinematography. Your role is to craft vivid, cinematographic video prompts characterized "
    "by authentic photochemical grain, wide organic dynamic range with soft highlight rolloff, and rich "
    "natural skin tones. Animate with gentle handheld camera drift or smooth fluid-head tripod pans, "
    "natural optical motion blur, and lifelike human and environmental physics. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_ANALOG_FILM = (
    "Render the video in authentic 35mm motion picture film cinematography. Depict the scene with "
    "authentic photochemical grain, wide organic dynamic range with soft highlight rolloff, and rich "
    "natural skin tones. Animate with gentle handheld camera drift or smooth fluid-head tripod pans, "
    "natural optical motion blur, and lifelike human and environmental physics. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_ANALOG_FILM = (
    "35mm motion picture film cinematography aesthetic. Authentic photochemical grain, wide organic "
    "dynamic range with soft highlight rolloff, and rich natural skin tones. Gentle handheld camera drift, "
    "smooth fluid-head tripod pans, natural optical motion blur, and lifelike human movement."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_AVAILABLE_LIGHT = (
    "You are an expert video director and cinematography prompt engineer specializing in natural "
    "available-light cinematography. Your role is to craft vivid, cinematographic video prompts "
    "characterized by wide aperture shallow focus, realistic exposure balanced against practical room "
    "lights or windows, deep natural shadows, and atmospheric low-light bokeh. Animate with subtle organic "
    "camera breathing, naturalistic subject movement through shifting light pools, and realistic focus "
    "pulls between focal planes. Describe the scene directly with concrete actions, camera movement, and "
    "stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_AVAILABLE_LIGHT = (
    "Render the video in authentic natural available-light cinematography. Depict the scene with wide "
    "aperture shallow focus, realistic exposure balanced against practical room lights or windows, deep "
    "natural shadows, and atmospheric low-light bokeh. Animate with subtle organic camera breathing, "
    "naturalistic subject movement through shifting light pools, and realistic focus pulls between focal "
    "planes. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_AVAILABLE_LIGHT = (
    "Natural available-light cinematography aesthetic. Wide aperture shallow focus, realistic exposure "
    "balanced against practical room lights or windows, deep natural shadows, and atmospheric low-light "
    "bokeh. Subtle organic camera breathing, naturalistic subject movement through shifting light pools, "
    "and realistic focus pulls."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_CCD_COMPACT_DIGITAL = (
    "You are an expert video director and cinematography prompt engineer specializing in early 2000s "
    "compact digital camcorder video. Your role is to craft vivid, cinematographic video prompts "
    "characterized by CCD sensor highlight blooming, punchy saturated primaries, and fine video sensor "
    "noise in midtones and shadows. Animate with casual handheld camera jitter, snappy automatic zoom "
    "transitions, and authentic consumer video shutter motion. Describe the scene directly with concrete "
    "actions, camera movement, and stylistic continuity without conversational filler or quality "
    "buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_CCD_COMPACT_DIGITAL = (
    "Render the video in authentic early 2000s compact digital camcorder video. Depict the scene with CCD "
    "sensor highlight blooming, punchy saturated primaries, and fine video sensor noise in midtones and "
    "shadows. Animate with casual handheld camera jitter, snappy automatic zoom transitions, and authentic "
    "consumer video shutter motion. Maintain spatial continuity, consistent lighting interaction, and "
    "coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_CCD_COMPACT_DIGITAL = (
    "Early 2000s compact digital camcorder video aesthetic. CCD sensor highlight blooming, punchy "
    "saturated primaries, and fine video sensor noise in midtones and shadows. Casual handheld camera "
    "jitter, snappy automatic zoom transitions, and authentic consumer video shutter motion."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_DIRECT_FLASH = (
    "You are an expert video director and cinematography prompt engineer specializing in direct on-camera "
    "flash cinematography. Your role is to craft vivid, cinematographic video prompts characterized by "
    "hard punchy frontal light, sharp dark drop shadows immediately behind subjects, crisp near-field "
    "illumination, and dark falloff in backgrounds. Animate with high-energy snapshot video motion, "
    "subjects moving through bright flash exposure, and rapid camera following. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_DIRECT_FLASH = (
    "Render the video in authentic direct on-camera flash cinematography. Depict the scene with hard "
    "punchy frontal light, sharp dark drop shadows immediately behind subjects, crisp near-field "
    "illumination, and dark falloff in backgrounds. Animate with high-energy snapshot video motion, "
    "subjects moving through bright flash exposure, and rapid camera following. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_DIRECT_FLASH = (
    "Direct on-camera flash cinematography aesthetic. Hard punchy frontal light, sharp dark drop shadows "
    "immediately behind subjects, crisp near-field illumination, and dark falloff in backgrounds. "
    "High-energy snapshot video motion, subjects moving through bright flash illumination, and rapid "
    "camera following."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_EARLY_2000S_ANALOG = (
    "You are an expert video director and cinematography prompt engineer specializing in consumer 35mm "
    "point-and-shoot snapshot video. Your role is to craft vivid, cinematographic video prompts "
    "characterized by warm minilab print color grading, moderate film grain, and nostalgic casual snapshot "
    "tones. Animate with natural unchoreographed subject movement, casual handheld camera operation, and "
    "authentic home-movie motion cadence. Describe the scene directly with concrete actions, camera "
    "movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_EARLY_2000S_ANALOG = (
    "Render the video in authentic consumer 35mm point-and-shoot snapshot video. Depict the scene with "
    "warm minilab print color grading, moderate film grain, and nostalgic casual snapshot tones. Animate "
    "with natural unchoreographed subject movement, casual handheld camera operation, and authentic "
    "home-movie motion cadence. Maintain spatial continuity, consistent lighting interaction, and coherent "
    "stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_EARLY_2000S_ANALOG = (
    "Consumer 35mm point-and-shoot snapshot video aesthetic. Warm minilab print color grading, moderate "
    "film grain, and nostalgic casual snapshot tones. Natural unchoreographed subject movement, casual "
    "handheld camera operation, and authentic home-movie motion cadence."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_FINE_ART_BLACK_AND_WHITE = (
    "You are an expert video director and cinematography prompt engineer specializing in fine-art "
    "monochrome cinematography. Your role is to craft vivid, cinematographic video prompts characterized "
    "by rich silver gelatin tonal gradation, deep velvety blacks, luminous midtones, and crisp optical "
    "sharpness. Animate with elegant slow camera movement, poetic subject movement, and dramatic interplay "
    "of light and shadow across moving surfaces. Describe the scene directly with concrete actions, camera "
    "movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_FINE_ART_BLACK_AND_WHITE = (
    "Render the video in authentic fine-art monochrome cinematography. Depict the scene with rich silver "
    "gelatin tonal gradation, deep velvety blacks, luminous midtones, and crisp optical sharpness. Animate "
    "with elegant slow camera movement, poetic subject movement, and dramatic interplay of light and "
    "shadow across moving surfaces. Maintain spatial continuity, consistent lighting interaction, and "
    "coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_FINE_ART_BLACK_AND_WHITE = (
    "Fine-art monochrome cinematography aesthetic. Rich silver gelatin tonal gradation, deep velvety "
    "blacks, luminous midtones, and crisp optical sharpness. Elegant slow camera movement, poetic subject "
    "movement, and dramatic interplay of light and shadow across moving surfaces."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_FULL_FRAME_DIGITAL = (
    "You are an expert video director and cinematography prompt engineer specializing in modern high-end "
    "full-frame digital cinematography. Your role is to craft vivid, cinematographic video prompts "
    "characterized by pristine optical sharpness, cinematic shallow depth of field, balanced exposure "
    "latitude, and clean color science. Animate with smooth gimbal-stabilized tracking shots, controlled "
    "cinematic camera dollies, lifelike physical momentum, and natural motion blur. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_FULL_FRAME_DIGITAL = (
    "Render the video in authentic modern high-end full-frame digital cinematography. Depict the scene "
    "with pristine optical sharpness, cinematic shallow depth of field, balanced exposure latitude, and "
    "clean color science. Animate with smooth gimbal-stabilized tracking shots, controlled cinematic "
    "camera dollies, lifelike physical momentum, and natural motion blur. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_FULL_FRAME_DIGITAL = (
    "Modern high-end full-frame digital cinematography aesthetic. Pristine optical sharpness, cinematic "
    "shallow depth of field, balanced exposure latitude, and clean natural color science. Smooth "
    "gimbal-stabilized tracking shots, controlled cinematic camera dollies, lifelike physical momentum, "
    "and natural motion blur."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_KODACHROME = (
    "You are an expert video director and cinematography prompt engineer specializing in vintage "
    "Kodachrome color motion picture film (1950s-1970s). Your role is to craft vivid, cinematographic "
    "video prompts characterized by deeply saturated primary colors, warm golden highlight cast, crisp "
    "photochemical grain, and distinctive archival film weave. Animate with period-authentic cinematic "
    "camera operation, natural human movement, and gentle film gate weave across frames. Describe the "
    "scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_KODACHROME = (
    "Render the video in authentic vintage Kodachrome color motion picture film (1950s-1970s). Depict the "
    "scene with deeply saturated primary colors, warm golden highlight cast, crisp photochemical grain, "
    "and distinctive archival film weave. Animate with period-authentic cinematic camera operation, "
    "natural human movement, and gentle film gate weave across frames. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_KODACHROME = (
    "Vintage Kodachrome color motion picture film aesthetic (1950s-1970s). Deeply saturated primary "
    "colors, warm golden highlight cast, crisp photochemical grain, and distinctive archival film weave. "
    "Period-authentic cinematic camera operation, natural human movement, and gentle film gate weave "
    "across frames."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_MEDIUM_FORMAT = (
    "You are an expert video director and cinematography prompt engineer specializing in medium-format "
    "motion cinematography. Your role is to craft vivid, cinematographic video prompts characterized by "
    "breathtaking subject separation, ultra-shallow depth of field, creamy background bokeh, and luminous "
    "tonal transitions. Animate with slow cinematic focus pulls, majestic camera tracking, and deliberate "
    "subject movement emphasizing dimensional depth. Describe the scene directly with concrete actions, "
    "camera movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_MEDIUM_FORMAT = (
    "Render the video in authentic medium-format motion cinematography. Depict the scene with breathtaking "
    "subject separation, ultra-shallow depth of field, creamy background bokeh, and luminous tonal "
    "transitions. Animate with slow cinematic focus pulls, majestic camera tracking, and deliberate "
    "subject movement emphasizing dimensional depth. Maintain spatial continuity, consistent lighting "
    "interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_MEDIUM_FORMAT = (
    "Medium-format motion cinematography aesthetic. Breathtaking subject separation, ultra-shallow depth "
    "of field, creamy background bokeh, and luminous tonal transitions. Slow cinematic focus pulls, "
    "majestic camera tracking, and deliberate subject movement emphasizing dimensional depth."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_PHOTOREALISM = (
    "You are an expert video director and cinematography prompt engineer specializing in documentary-grade "
    "photorealistic live action. Your role is to craft vivid, cinematographic video prompts characterized "
    "by natural human movement, physically accurate environmental lighting and reflections, and "
    "true-to-life optical rendering. Animate with natural human movement displaying realistic inertia, "
    "authentic environmental physical dynamics, and subtle organic camera breathing. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_PHOTOREALISM = (
    "Render the video in authentic documentary-grade photorealistic live action. Depict the scene with "
    "natural human movement, physically accurate environmental lighting and reflections, and true-to-life "
    "optical rendering. Animate with natural human movement displaying realistic inertia, authentic "
    "environmental physical dynamics, and subtle organic camera breathing. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_PHOTOREALISM = (
    "Documentary-grade photorealistic live action aesthetic. Natural human movement, physically accurate "
    "environmental lighting and reflections, and true-to-life optical rendering. Natural human movement "
    "with realistic inertia, authentic environmental physical dynamics, and subtle organic camera "
    "breathing."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_POLAROID = (
    "You are an expert video director and cinematography prompt engineer specializing in vintage Polaroid "
    "instant film video. Your role is to craft vivid, cinematographic video prompts characterized by warm "
    "faded pastels, soft vignetted edges, gentle organic light leaks, and subtle exposure flutter. Animate "
    "with gentle handheld camera movement, soft motion blur, nostalgic unchoreographed human actions, and "
    "organic frame jitter. Describe the scene directly with concrete actions, camera movement, and "
    "stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_POLAROID = (
    "Render the video in authentic vintage Polaroid instant film video. Depict the scene with warm faded "
    "pastels, soft vignetted edges, gentle organic light leaks, and subtle exposure flutter. Animate with "
    "gentle handheld camera movement, soft motion blur, nostalgic unchoreographed human actions, and "
    "organic frame jitter. Maintain spatial continuity, consistent lighting interaction, and coherent "
    "stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_POLAROID = (
    "Vintage Polaroid instant film video aesthetic. Warm faded pastels, soft vignetted edges, gentle "
    "organic light leaks, and subtle exposure flutter. Gentle handheld camera movement, soft motion blur, "
    "nostalgic unchoreographed human actions, and organic frame jitter."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_SMARTPHONE = (
    "You are an expert video director and cinematography prompt engineer specializing in modern casual "
    "mobile video. Your role is to craft vivid, cinematographic video prompts characterized by wide-angle "
    "computational HDR rendering, crisp deep focus, and natural handheld movement with electronic "
    "stabilization artifacts. Animate with everyday handheld point-of-view motion, quick mobile pans, and "
    "casual natural subject interactions. Describe the scene directly with concrete actions, camera "
    "movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_SMARTPHONE = (
    "Render the video in authentic modern casual mobile video. Depict the scene with wide-angle "
    "computational HDR rendering, crisp deep focus, and natural handheld movement with electronic "
    "stabilization artifacts. Animate with everyday handheld point-of-view motion, quick mobile pans, and "
    "casual natural subject interactions. Maintain spatial continuity, consistent lighting interaction, "
    "and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_SMARTPHONE = (
    "Modern casual mobile video aesthetic. Wide-angle computational HDR rendering, crisp deep focus, and "
    "natural handheld camera movement with subtle electronic stabilization artifacts. Everyday "
    "point-of-view perspective, quick mobile pans, and casual natural subject interactions."
)

SYSTEM_MESSAGE_STYLE_PHOTOGRAPHY_STUDIO_STROBE = (
    "You are an expert video director and cinematography prompt engineer specializing in commercial studio "
    "cinematography. Your role is to craft vivid, cinematographic video prompts characterized by sculpted "
    "multi-point key, fill, and rim lighting, pristine specular highlights, deep edge-to-edge sharpness, "
    "and controlled clean backgrounds. Animate with controlled model movement, smooth mechanical studio "
    "dolly tracking, and precise lighting reflections shifting across polished surfaces. Describe the "
    "scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PHOTOGRAPHY_STUDIO_STROBE = (
    "Render the video in authentic commercial studio cinematography. Depict the scene with sculpted "
    "multi-point key, fill, and rim lighting, pristine specular highlights, deep edge-to-edge sharpness, "
    "and controlled clean backgrounds. Animate with controlled model movement, smooth mechanical studio "
    "dolly tracking, and precise lighting reflections shifting across polished surfaces. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PHOTOGRAPHY_STUDIO_STROBE = (
    "Commercial studio cinematography aesthetic. Sculpted multi-point key, fill, and rim lighting, "
    "pristine specular highlights, deep edge-to-edge sharpness, and controlled clean backgrounds. "
    "Controlled model movement, smooth mechanical studio dolly tracking, and precise lighting reflections "
    "shifting across polished surfaces."
)

SYSTEM_MESSAGE_STYLE_PIXAR = (
    "You are an expert video director and cinematography prompt engineer specializing in Pixar Animation "
    "Studios 3D CGI feature film style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by appealing stylized character proportions, tactile PBR textures with subtle "
    "subsurface scattering, emotive micro-expressions, and rich cinematic volumetric lighting. Animate "
    "with highly expressive character acting, fluid physical comedy, sweeping cinematic camera staging, "
    "and lively secondary animation on cloth, hair, and props. Describe the scene directly with concrete "
    "actions, camera movement, and stylistic continuity without conversational filler or quality "
    "buzzwords."
)

INSTRUCT_PROMPT_STYLE_PIXAR = (
    "Render the video in authentic Pixar Animation Studios 3D CGI feature film style. Depict the scene "
    "with appealing stylized character proportions, tactile PBR textures with subtle subsurface "
    "scattering, emotive micro-expressions, and rich cinematic volumetric lighting. Animate with highly "
    "expressive character acting, fluid physical comedy, sweeping cinematic camera staging, and lively "
    "secondary animation on cloth, hair, and props. Maintain spatial continuity, consistent lighting "
    "interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PIXAR = (
    "Pixar Animation Studios 3D CGI feature film aesthetic. Appealing stylized character proportions, "
    "tactile PBR textures with subtle subsurface scattering, emotive micro-expressions, and rich cinematic "
    "volumetric lighting. Highly expressive character acting, fluid physical comedy, sweeping cinematic "
    "camera staging, and lively secondary animation on cloth, hair, and props."
)

SYSTEM_MESSAGE_STYLE_PIXEL_ART = (
    "You are an expert video director and cinematography prompt engineer specializing in authentic retro "
    "pixel art animation. Your role is to craft vivid, cinematographic video prompts characterized by "
    "limited color palettes, hand-drawn sprite frames on a stepped pixel grid, visible pixel clusters, "
    "classic walk cycles, and CRT scanline warmth. Animate with stepped frame-by-frame sprite animation, "
    "classic 2D platformer movement physics, and parallax scrolling background layers. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PIXEL_ART = (
    "Render the video in authentic authentic retro pixel art animation. Depict the scene with limited "
    "color palettes, hand-drawn sprite frames on a stepped pixel grid, visible pixel clusters, classic "
    "walk cycles, and CRT scanline warmth. Animate with stepped frame-by-frame sprite animation, classic "
    "2D platformer movement physics, and parallax scrolling background layers. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PIXEL_ART = (
    "Authentic retro pixel art animation aesthetic. Limited color palettes, hand-drawn sprite frames on a "
    "stepped pixel grid, visible pixel clusters, classic walk cycles, and CRT scanline warmth. Stepped "
    "frame-by-frame sprite animation, classic 2D platformer movement physics, and parallax scrolling "
    "background layers."
)

SYSTEM_MESSAGE_STYLE_POKEMON = (
    "You are an expert video director and cinematography prompt engineer specializing in Pokémon "
    "cel-shaded adventure anime style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by bright vibrant color palettes, clean black outlines, energetic creature animation, "
    "and clear sunny outdoor lighting. Animate with dynamic creature battle choreography, energetic "
    "running and leaping, bright elemental particle bursts, and sweeping anime camera angles. Describe the "
    "scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_POKEMON = (
    "Render the video in authentic Pokémon cel-shaded adventure anime style. Depict the scene with bright "
    "vibrant color palettes, clean black outlines, energetic creature animation, and clear sunny outdoor "
    "lighting. Animate with dynamic creature battle choreography, energetic running and leaping, bright "
    "elemental particle bursts, and sweeping anime camera angles. Maintain spatial continuity, consistent "
    "lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_POKEMON = (
    "Pokémon cel-shaded adventure anime aesthetic. Bright vibrant color palettes, clean black outlines, "
    "energetic creature animation, and clear sunny outdoor lighting. Dynamic creature battle choreography, "
    "energetic running and leaping, bright elemental particle bursts, and sweeping anime camera angles."
)

SYSTEM_MESSAGE_STYLE_PRESERVE_IDENTITY = (
    "You are an expert video director and cinematography prompt engineer specializing in "
    "identity-consistent video generation. Your role is to craft vivid, cinematographic video prompts "
    "characterized by strictly maintaining the subject's distinct facial likeness, structural proportions, "
    "and identifying features across dynamic camera motion and environmental lighting. Animate with "
    "consistent facial and anatomical tracking through head rotations, expressive movements, and changing "
    "camera perspectives while integrating naturally into the scene. Describe the scene directly with "
    "concrete actions, camera movement, and stylistic continuity without conversational filler or quality "
    "buzzwords."
)

INSTRUCT_PROMPT_STYLE_PRESERVE_IDENTITY = (
    "Render the video in authentic identity-consistent video generation. Depict the scene with strictly "
    "maintaining the subject's distinct facial likeness, structural proportions, and identifying features "
    "across dynamic camera motion and environmental lighting. Animate with consistent facial and "
    "anatomical tracking through head rotations, expressive movements, and changing camera perspectives "
    "while integrating naturally into the scene. Maintain spatial continuity, consistent lighting "
    "interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PRESERVE_IDENTITY = (
    "Identity-consistent video generation aesthetic. Strictly maintains the subject's distinct facial "
    "likeness, structural proportions, and identifying features across dynamic camera motion and "
    "environmental lighting. Natural facial and anatomical tracking through head turns, expressive "
    "gestures, and shifting camera perspectives."
)

SYSTEM_MESSAGE_STYLE_PRESERVE_STYLE = (
    "You are an expert video director and cinematography prompt engineer specializing in style-consistent "
    "video generation. Your role is to craft vivid, cinematographic video prompts characterized by "
    "strictly preserving the established artistic medium, rendering technique, color palette, and textural "
    "aesthetic across all frames and movement. Animate with consistent artistic rendering, surface "
    "textures, and line quality across all dynamic actions, lighting shifts, and camera moves. Describe "
    "the scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PRESERVE_STYLE = (
    "Render the video in authentic style-consistent video generation. Depict the scene with strictly "
    "preserving the established artistic medium, rendering technique, color palette, and textural "
    "aesthetic across all frames and movement. Animate with consistent artistic rendering, surface "
    "textures, and line quality across all dynamic actions, lighting shifts, and camera moves. Maintain "
    "spatial continuity, consistent lighting interaction, and coherent stylistic physics throughout the "
    "sequence."
)

BONUS_PROMPT_STYLE_PRESERVE_STYLE = (
    "Style-consistent video generation aesthetic. Strictly preserves the established artistic medium, "
    "rendering technique, color palette, and textural aesthetic across all frames and movement. Maintains "
    "coherent surface textures, line quality, and stylistic lighting across all dynamic actions and camera "
    "moves."
)

SYSTEM_MESSAGE_STYLE_PS1 = (
    "You are an expert video director and cinematography prompt engineer specializing in Sony PlayStation "
    "1 (PS1) retro 32-bit 3D video game graphics. Your role is to craft vivid, cinematographic video "
    "prompts characterized by low-polygon meshes, affine texture warping, wobbling vertex jitter, "
    "pixelated textures, and flat-shaded polygon lighting. Animate with rigid retro 3D game animation "
    "cycles, low-framerate character movement, and classic third-person fixed camera angles with early "
    "console tracking. Describe the scene directly with concrete actions, camera movement, and stylistic "
    "continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_PS1 = (
    "Render the video in authentic Sony PlayStation 1 (PS1) retro 32-bit 3D video game graphics. Depict "
    "the scene with low-polygon meshes, affine texture warping, wobbling vertex jitter, pixelated "
    "textures, and flat-shaded polygon lighting. Animate with rigid retro 3D game animation cycles, "
    "low-framerate character movement, and classic third-person fixed camera angles with early console "
    "tracking. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic "
    "physics throughout the sequence."
)

BONUS_PROMPT_STYLE_PS1 = (
    "Sony PlayStation 1 (PS1) retro 32-bit 3D video game aesthetic. Low-polygon meshes, affine texture "
    "warping, wobbling vertex jitter, pixelated textures, and flat-shaded polygon lighting. Rigid retro 3D "
    "game animation cycles, low-framerate character movement, and classic third-person fixed camera angles "
    "with early console tracking."
)

SYSTEM_MESSAGE_STYLE_RETRO_GAME = (
    "You are an expert video director and cinematography prompt engineer specializing in 8-bit / 16-bit "
    "arcade video game animation. Your role is to craft vivid, cinematographic video prompts characterized "
    "by limited indexed color palettes, pixelated sprites with clean silhouette animation, stepped sprite "
    "movement, and retro gaming CRT aesthetics. Animate with stepped sprite walk and attack cycles, "
    "classic arcade side-scroller motion, and snappy retro physics. Describe the scene directly with "
    "concrete actions, camera movement, and stylistic continuity without conversational filler or quality "
    "buzzwords."
)

INSTRUCT_PROMPT_STYLE_RETRO_GAME = (
    "Render the video in authentic 8-bit / 16-bit arcade video game animation. Depict the scene with "
    "limited indexed color palettes, pixelated sprites with clean silhouette animation, stepped sprite "
    "movement, and retro gaming CRT aesthetics. Animate with stepped sprite walk and attack cycles, "
    "classic arcade side-scroller motion, and snappy retro physics. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_RETRO_GAME = (
    "8-bit / 16-bit arcade video game animation aesthetic. Limited indexed color palettes, pixelated "
    "sprites with clean silhouette animation, stepped sprite movement, and retro gaming CRT aesthetics. "
    "Stepped sprite walk and attack cycles, classic arcade side-scroller motion, and snappy retro physics."
)

SYSTEM_MESSAGE_STYLE_RICK_AND_MORTY = (
    "You are an expert video director and cinematography prompt engineer specializing in Rick and Morty "
    "animated cartoon style. Your role is to craft vivid, cinematographic video prompts characterized by "
    "scribbled pupil irises, relaxed rounded anatomy, clean uneven black outlines, glowing portal effects, "
    "and eccentric interdimensional sci-fi physics. Animate with snappy dialogue gestures, fluid "
    "interdimensional portal transitions, eccentric sci-fi gadget interactions, and chaotic physical "
    "comedy. Describe the scene directly with concrete actions, camera movement, and stylistic continuity "
    "without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_RICK_AND_MORTY = (
    "Render the video in authentic Rick and Morty animated cartoon style. Depict the scene with scribbled "
    "pupil irises, relaxed rounded anatomy, clean uneven black outlines, glowing portal effects, and "
    "eccentric interdimensional sci-fi physics. Animate with snappy dialogue gestures, fluid "
    "interdimensional portal transitions, eccentric sci-fi gadget interactions, and chaotic physical "
    "comedy. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_RICK_AND_MORTY = (
    "Rick and Morty animated cartoon aesthetic. Scribbled pupil irises, relaxed rounded anatomy, clean "
    "uneven black outlines, glowing portal effects, and eccentric interdimensional sci-fi physics. Snappy "
    "dialogue gestures, fluid interdimensional portal transitions, eccentric sci-fi gadget interactions, "
    "and chaotic physical comedy."
)

SYSTEM_MESSAGE_STYLE_RPG = (
    "You are an expert video director and cinematography prompt engineer specializing in classic fantasy "
    "RPG game engine cinematic aesthetic. Your role is to craft vivid, cinematographic video prompts "
    "characterized by rich fantasy character classes, atmospheric environmental fog, real-time particle "
    "and spell effects, and dynamic follow camera staging. Animate with responsive fantasy adventure "
    "movement, dynamic spellcasting and swordplay choreography, atmospheric weather physics, and cinematic "
    "over-the-shoulder tracking. Describe the scene directly with concrete actions, camera movement, and "
    "stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_RPG = (
    "Render the video in authentic classic fantasy RPG game engine cinematic aesthetic. Depict the scene "
    "with rich fantasy character classes, atmospheric environmental fog, real-time particle and spell "
    "effects, and dynamic follow camera staging. Animate with responsive fantasy adventure movement, "
    "dynamic spellcasting and swordplay choreography, atmospheric weather physics, and cinematic "
    "over-the-shoulder tracking. Maintain spatial continuity, consistent lighting interaction, and "
    "coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_RPG = (
    "Classic fantasy RPG game engine cinematic aesthetic. Rich fantasy character classes, atmospheric "
    "environmental fog, real-time particle and spell effects, and dynamic follow camera staging. "
    "Responsive fantasy adventure movement, dynamic spellcasting and swordplay choreography, atmospheric "
    "weather physics, and cinematic over-the-shoulder tracking."
)

SYSTEM_MESSAGE_STYLE_SAILOR_MOON = (
    "You are an expert video director and cinematography prompt engineer specializing in Sailor Moon 1990s "
    "magical girl anime style. Your role is to craft vivid, cinematographic video prompts characterized by "
    "soft romantic pastel color palettes, sparkling glowing magical particles, elegant elongated character "
    "proportions, and dramatic hair and ribbon wind physics. Animate with lyrical transformation "
    "flourishes, dynamic wind-blown hair and ribbons, sparkling optical particle bursts, and dramatic "
    "low-angle heroic poses. Describe the scene directly with concrete actions, camera movement, and "
    "stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_SAILOR_MOON = (
    "Render the video in authentic Sailor Moon 1990s magical girl anime style. Depict the scene with soft "
    "romantic pastel color palettes, sparkling glowing magical particles, elegant elongated character "
    "proportions, and dramatic hair and ribbon wind physics. Animate with lyrical transformation "
    "flourishes, dynamic wind-blown hair and ribbons, sparkling optical particle bursts, and dramatic "
    "low-angle heroic poses. Maintain spatial continuity, consistent lighting interaction, and coherent "
    "stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_SAILOR_MOON = (
    "Sailor Moon 1990s magical girl anime aesthetic. Soft romantic pastel color palettes, sparkling "
    "glowing magical particles, elegant elongated character proportions, and dramatic hair and ribbon wind "
    "physics. Lyrical transformation flourishes, dynamic wind-blown hair and ribbons, sparkling optical "
    "particle bursts, and dramatic low-angle heroic poses."
)

SYSTEM_MESSAGE_STYLE_SIMS = (
    "You are an expert video director and cinematography prompt engineer specializing in The Sims 3D life "
    "simulation video game aesthetic. Your role is to craft vivid, cinematographic video prompts "
    "characterized by smooth character models, cheerful saturated colors, clean domestic environments, "
    "casual emotive idle animations, and soft ambient lighting. Animate with expressive emotional "
    "gestures, distinctive life-sim idle animations, cheerful everyday domestic movement, and clean "
    "isometric or third-person game camera framing. Describe the scene directly with concrete actions, "
    "camera movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_SIMS = (
    "Render the video in authentic The Sims 3D life simulation video game aesthetic. Depict the scene with "
    "smooth character models, cheerful saturated colors, clean domestic environments, casual emotive idle "
    "animations, and soft ambient lighting. Animate with expressive emotional gestures, distinctive "
    "life-sim idle animations, cheerful everyday domestic movement, and clean isometric or third-person "
    "game camera framing. Maintain spatial continuity, consistent lighting interaction, and coherent "
    "stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_SIMS = (
    "The Sims 3D life simulation video game aesthetic. Smooth character models, cheerful saturated colors, "
    "clean domestic environments, casual emotive idle animations, and soft ambient lighting. Expressive "
    "emotional gestures, distinctive life-sim idle animations, cheerful everyday domestic movement, and "
    "clean isometric or third-person game camera framing."
)

SYSTEM_MESSAGE_STYLE_SKYRIM = (
    "You are an expert video director and cinematography prompt engineer specializing in The Elder "
    "Scrolls: Skyrim medieval fantasy RPG game engine aesthetic. Your role is to craft vivid, "
    "cinematographic video prompts characterized by Nordic atmospheric mist, weathered iron and fur armor, "
    "rugged stone and pine environments, and cold naturalistic lighting. Animate with grounded, "
    "weight-bearing character movement, realistic weapon swings with physical inertia, atmospheric "
    "drifting snow and fog, and third-person fantasy RPG camera follow. Describe the scene directly with "
    "concrete actions, camera movement, and stylistic continuity without conversational filler or quality "
    "buzzwords."
)

INSTRUCT_PROMPT_STYLE_SKYRIM = (
    "Render the video in authentic The Elder Scrolls: Skyrim medieval fantasy RPG game engine aesthetic. "
    "Depict the scene with Nordic atmospheric mist, weathered iron and fur armor, rugged stone and pine "
    "environments, and cold naturalistic lighting. Animate with grounded, weight-bearing character "
    "movement, realistic weapon swings with physical inertia, atmospheric drifting snow and fog, and "
    "third-person fantasy RPG camera follow. Maintain spatial continuity, consistent lighting interaction, "
    "and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_SKYRIM = (
    "The Elder Scrolls: Skyrim medieval fantasy RPG game engine aesthetic. Nordic atmospheric mist, "
    "weathered iron and fur armor, rugged stone and pine environments, and cold naturalistic lighting. "
    "Grounded, weight-bearing character movement, realistic weapon swings with physical inertia, "
    "atmospheric drifting snow and fog, and third-person fantasy RPG camera follow."
)

SYSTEM_MESSAGE_STYLE_SOUTH_PARK = (
    "You are an expert video director and cinematography prompt engineer specializing in South Park cutout "
    "construction paper animation style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by jerky limited frame rate, flat paper textures with visible cutout edges, simple "
    "geometric character silhouettes, and primitive hinged limb movement. Animate with distinctive jerky "
    "stop-motion frame skips, paper cutout hopping and skipping, primitive hinged limb articulation, and "
    "deadpan comic timing. Describe the scene directly with concrete actions, camera movement, and "
    "stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_SOUTH_PARK = (
    "Render the video in authentic South Park cutout construction paper animation style. Depict the scene "
    "with jerky limited frame rate, flat paper textures with visible cutout edges, simple geometric "
    "character silhouettes, and primitive hinged limb movement. Animate with distinctive jerky stop-motion "
    "frame skips, paper cutout hopping and skipping, primitive hinged limb articulation, and deadpan comic "
    "timing. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_SOUTH_PARK = (
    "South Park cutout construction paper animation aesthetic. Jerky limited frame rate, flat paper "
    "textures with visible cutout edges, simple geometric character silhouettes, and primitive hinged limb "
    "movement. Distinctive jerky stop-motion frame skips, paper cutout hopping and skipping, primitive "
    "hinged limb articulation, and deadpan comic timing."
)

SYSTEM_MESSAGE_STYLE_STAINED_GLASS = (
    "You are an expert video director and cinematography prompt engineer specializing in crafted stained "
    "glass window art style. Your role is to craft vivid, cinematographic video prompts characterized by "
    "luminous translucent colored glass segments, bold dark lead cames framing all forms, radiant "
    "backlighting passing through colored panes, and jewel-toned illumination. Animate with luminous "
    "shifting light beams passing through colored glass, subtle caustic light patterns moving across "
    "surfaces, and majestic slow camera sweeps. Describe the scene directly with concrete actions, camera "
    "movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_STAINED_GLASS = (
    "Render the video in authentic crafted stained glass window art style. Depict the scene with luminous "
    "translucent colored glass segments, bold dark lead cames framing all forms, radiant backlighting "
    "passing through colored panes, and jewel-toned illumination. Animate with luminous shifting light "
    "beams passing through colored glass, subtle caustic light patterns moving across surfaces, and "
    "majestic slow camera sweeps. Maintain spatial continuity, consistent lighting interaction, and "
    "coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_STAINED_GLASS = (
    "Crafted stained glass window art aesthetic. Luminous translucent colored glass segments, bold dark "
    "lead cames framing all forms, radiant backlighting passing through colored panes, and jewel-toned "
    "illumination. Shifting luminous light beams passing through colored glass, caustic light patterns "
    "moving across surfaces, and majestic slow camera sweeps."
)

SYSTEM_MESSAGE_STYLE_STEAMPUNK = (
    "You are an expert video director and cinematography prompt engineer specializing in Victorian "
    "steampunk industrial art style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by polished brass and copper clockwork gears, pressurized steam vents, exposed rivets, "
    "warm sepia-toned gaslight illumination, and ornate Victorian mechanical design. Animate with churning "
    "clockwork gear rotations, rhythmic venting steam bursts, realistic mechanical articulation, and "
    "atmospheric gaslight flicker. Describe the scene directly with concrete actions, camera movement, and "
    "stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_STEAMPUNK = (
    "Render the video in authentic Victorian steampunk industrial art style. Depict the scene with "
    "polished brass and copper clockwork gears, pressurized steam vents, exposed rivets, warm sepia-toned "
    "gaslight illumination, and ornate Victorian mechanical design. Animate with churning clockwork gear "
    "rotations, rhythmic venting steam bursts, realistic mechanical articulation, and atmospheric gaslight "
    "flicker. Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_STEAMPUNK = (
    "Victorian steampunk industrial art aesthetic. Polished brass and copper clockwork gears, pressurized "
    "steam vents, exposed rivets, warm sepia-toned gaslight illumination, and ornate mechanical design. "
    "Churning clockwork gear rotations, rhythmic venting steam bursts, realistic mechanical articulation, "
    "and atmospheric gaslight flicker."
)

SYSTEM_MESSAGE_STYLE_SYNTHWAVE = (
    "You are an expert video director and cinematography prompt engineer specializing in 1980s "
    "retro-futuristic synthwave digital art style. Your role is to craft vivid, cinematographic video "
    "prompts characterized by glowing neon grid horizons, reflective chrome typography, vibrant magenta "
    "and cyan sunset skies, laser beams, and a driving retro-electronic visual rhythm. Animate with "
    "driving forward tracking motion over endless wireframe grids, sweeping neon light reflections, and "
    "rhythmic pulsating atmospheric glow. Describe the scene directly with concrete actions, camera "
    "movement, and stylistic continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_SYNTHWAVE = (
    "Render the video in authentic 1980s retro-futuristic synthwave digital art style. Depict the scene "
    "with glowing neon grid horizons, reflective chrome typography, vibrant magenta and cyan sunset skies, "
    "laser beams, and a driving retro-electronic visual rhythm. Animate with driving forward tracking "
    "motion over endless wireframe grids, sweeping neon light reflections, and rhythmic pulsating "
    "atmospheric glow. Maintain spatial continuity, consistent lighting interaction, and coherent "
    "stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_SYNTHWAVE = (
    "1980s retro-futuristic synthwave digital art aesthetic. Glowing neon grid horizons, reflective chrome "
    "typography, vibrant magenta and cyan sunset skies, laser beams, and a driving retro-electronic visual "
    "rhythm. Driving forward tracking motion over endless wireframe grids, sweeping neon light "
    "reflections, and rhythmic pulsating atmospheric glow."
)

SYSTEM_MESSAGE_STYLE_VECTOR_EDIT = (
    "You are an expert video director and cinematography prompt engineer specializing in clean scalar "
    "vector graphic illustration style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by mathematically precise bezier curves, flat geometric color fills, crisp graphic "
    "silhouette contours, and zero pixel-based texture. Animate with fluid geometric morphing, smooth "
    "flat-color vector pans, clean rotational pivots, and crisp graphic motion. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_VECTOR_EDIT = (
    "Render the video in authentic clean scalar vector graphic illustration style. Depict the scene with "
    "mathematically precise bezier curves, flat geometric color fills, crisp graphic silhouette contours, "
    "and zero pixel-based texture. Animate with fluid geometric morphing, smooth flat-color vector pans, "
    "clean rotational pivots, and crisp graphic motion. Maintain spatial continuity, consistent lighting "
    "interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_VECTOR_EDIT = (
    "Clean scalar vector graphic illustration aesthetic. Mathematically precise bezier curves, flat "
    "geometric color fills, crisp graphic silhouette contours, and zero pixel-based texture. Fluid "
    "geometric morphing, smooth flat-color vector pans, clean rotational pivots, and crisp graphic motion."
)

SYSTEM_MESSAGE_STYLE_VECTOR_RAW = (
    "You are an expert video director and cinematography prompt engineer specializing in minimalist modern "
    "vector graphic art style. Your role is to craft vivid, cinematographic video prompts characterized by "
    "flat graphic shapes, bold minimalist color palettes, razor-sharp contour definition, and modern "
    "digital iconographic styling. Animate with snappy motion-graphic transitions, flat geometric camera "
    "panning, synchronized motion-graphic timing, and clean shape transformations. Describe the scene "
    "directly with concrete actions, camera movement, and stylistic continuity without conversational "
    "filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_VECTOR_RAW = (
    "Render the video in authentic minimalist modern vector graphic art style. Depict the scene with flat "
    "graphic shapes, bold minimalist color palettes, razor-sharp contour definition, and modern digital "
    "iconographic styling. Animate with snappy motion-graphic transitions, flat geometric camera panning, "
    "synchronized motion-graphic timing, and clean shape transformations. Maintain spatial continuity, "
    "consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_VECTOR_RAW = (
    "Minimalist modern vector graphic art aesthetic. Flat graphic shapes, bold minimalist color palettes, "
    "razor-sharp contour definition, and modern digital iconographic styling. Snappy motion-graphic "
    "transitions, flat geometric camera panning, synchronized motion-graphic timing, and clean shape "
    "transformations."
)

SYSTEM_MESSAGE_STYLE_VIDEO_GAME = (
    "You are an expert video director and cinematography prompt engineer specializing in modern "
    "third-person action video game capture. Your role is to craft vivid, cinematographic video prompts "
    "characterized by real-time game engine rendering, dynamic follow camera, responsive character "
    "movement, particle effects, and dynamic lighting. Animate with responsive action game character "
    "movement, dynamic over-the-shoulder follow camera, real-time particle effects, and physical collision "
    "responses. Describe the scene directly with concrete actions, camera movement, and stylistic "
    "continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_VIDEO_GAME = (
    "Render the video in authentic modern third-person action video game capture. Depict the scene with "
    "real-time game engine rendering, dynamic follow camera, responsive character movement, particle "
    "effects, and dynamic lighting. Animate with responsive action game character movement, dynamic "
    "over-the-shoulder follow camera, real-time particle effects, and physical collision responses. "
    "Maintain spatial continuity, consistent lighting interaction, and coherent stylistic physics "
    "throughout the sequence."
)

BONUS_PROMPT_STYLE_VIDEO_GAME = (
    "Modern third-person action video game aesthetic. Real-time game engine rendering, dynamic follow "
    "camera, responsive character movement, particle effects, and dynamic lighting. Responsive action game "
    "character movement, dynamic over-the-shoulder follow camera, real-time particle effects, and physical "
    "collision responses."
)

SYSTEM_MESSAGE_STYLE_WATERCOLOR = (
    "You are an expert video director and cinematography prompt engineer specializing in traditional "
    "fine-art watercolor painting style. Your role is to craft vivid, cinematographic video prompts "
    "characterized by translucent wet-on-wet color washes, organic pigment bleeding, visible cotton paper "
    "tooth, soft luminous edge diffusion, and fluid brushwork textures. Animate with organic fluid pigment "
    "blooms, soft watercolor wash transitions, atmospheric paper texture breathing, and lyrical painterly "
    "character movement. Describe the scene directly with concrete actions, camera movement, and stylistic "
    "continuity without conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_WATERCOLOR = (
    "Render the video in authentic traditional fine-art watercolor painting style. Depict the scene with "
    "translucent wet-on-wet color washes, organic pigment bleeding, visible cotton paper tooth, soft "
    "luminous edge diffusion, and fluid brushwork textures. Animate with organic fluid pigment blooms, "
    "soft watercolor wash transitions, atmospheric paper texture breathing, and lyrical painterly "
    "character movement. Maintain spatial continuity, consistent lighting interaction, and coherent "
    "stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_WATERCOLOR = (
    "Traditional fine-art watercolor painting aesthetic. Translucent wet-on-wet color washes, organic "
    "pigment bleeding, visible cotton paper tooth, soft luminous edge diffusion, and fluid brushwork "
    "textures. Organic fluid pigment blooms, soft watercolor wash transitions, atmospheric paper texture "
    "breathing, and lyrical painterly character movement."
)

SYSTEM_MESSAGE_STYLE_ZELDA_3D = (
    "You are an expert video director and cinematography prompt engineer specializing in The Legend of "
    "Zelda painterly cel-shaded 3D adventure game aesthetic (Nintendo). Your role is to craft vivid, "
    "cinematographic video prompts characterized by impressionistic watercolor-like textures, dynamic wind "
    "shader effects on foliage, warm golden-hour lighting, and sweeping cinematic adventure camera moves. "
    "Animate with heroic dynamic camera sweeps, responsive acrobatic movement, wind billowing through "
    "grass and clothing, and sparkling ancient technology particle effects. Describe the scene directly "
    "with concrete actions, camera movement, and stylistic continuity without conversational filler or "
    "quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_ZELDA_3D = (
    "Render the video in authentic The Legend of Zelda painterly cel-shaded 3D adventure game aesthetic "
    "(Nintendo). Depict the scene with impressionistic watercolor-like textures, dynamic wind shader "
    "effects on foliage, warm golden-hour lighting, and sweeping cinematic adventure camera moves. Animate "
    "with heroic dynamic camera sweeps, responsive acrobatic movement, wind billowing through grass and "
    "clothing, and sparkling ancient technology particle effects. Maintain spatial continuity, consistent "
    "lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_ZELDA_3D = (
    "The Legend of Zelda painterly cel-shaded 3D adventure game aesthetic (Nintendo). Impressionistic "
    "watercolor-like textures, dynamic wind shader effects on foliage, warm golden-hour lighting, and "
    "sweeping cinematic adventure camera moves. Heroic dynamic camera sweeps, responsive acrobatic "
    "movement, wind billowing through grass and clothing, and sparkling ancient technology particle "
    "effects."
)

SYSTEM_MESSAGE_STYLE_ZELDA_RETRO = (
    "You are an expert video director and cinematography prompt engineer specializing in classic 16-bit "
    "top-down action-adventure game graphics (The Legend of Zelda: A Link to the Past style). Your role is "
    "to craft vivid, cinematographic video prompts characterized by rich 16-bit pixel art tilesets, "
    "hand-drawn sprite walk and attack cycles, classic dungeon and overworld color palettes, and stepped "
    "cardinal direction motion. Animate with stepped 4-directional sprite walk cycles, snappy sword spin "
    "attacks, classic top-down scrolling camera movement, and retro 16-bit particle bursts. Describe the "
    "scene directly with concrete actions, camera movement, and stylistic continuity without "
    "conversational filler or quality buzzwords."
)

INSTRUCT_PROMPT_STYLE_ZELDA_RETRO = (
    "Render the video in authentic classic 16-bit top-down action-adventure game graphics (The Legend of "
    "Zelda: A Link to the Past style). Depict the scene with rich 16-bit pixel art tilesets, hand-drawn "
    "sprite walk and attack cycles, classic dungeon and overworld color palettes, and stepped cardinal "
    "direction motion. Animate with stepped 4-directional sprite walk cycles, snappy sword spin attacks, "
    "classic top-down scrolling camera movement, and retro 16-bit particle bursts. Maintain spatial "
    "continuity, consistent lighting interaction, and coherent stylistic physics throughout the sequence."
)

BONUS_PROMPT_STYLE_ZELDA_RETRO = (
    "Classic 16-bit top-down action-adventure game aesthetic (The Legend of Zelda: A Link to the Past "
    "style). Rich 16-bit pixel art tilesets, hand-drawn sprite walk and attack cycles, classic dungeon and "
    "overworld palettes, and stepped cardinal direction motion. Stepped 4-directional sprite walk cycles, "
    "snappy sword spin attacks, classic top-down scrolling camera movement, and retro 16-bit particle "
    "bursts."
)
