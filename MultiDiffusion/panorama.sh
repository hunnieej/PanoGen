#!/usr/bin/env bash
# run_panorama_sana_grid.sh
# 반복 실행 스            n=$((n+1))
            ofn="${OUTDIR}/pan_${n}_${pslug}_neg-${nslug}_H${H}_W${W}_S${steps}_seed${seed}"
            echo "[RUN $n] H=$H W=$W steps=$steps seed=$seed GPU=$GPU_NUM"
            echo "         prompt='${prompt}'"
            echo "         negative='${neg}'"
            echo "         output folder: $ofn"다양한 prompt/negative/H/W/steps/seed 조합으로 panorama_360.py 실행

set -euo pipefail

PY=python
SCRIPT=panorama_360.py

# ===== 설정 =====
SD_VERSION="2.1"          # "1.5", "2.0", "2.1" "SANA"
GPU_NUM=1                 # GPU device number to use (matching launch.json)
SAVE_INTERMEDIATE=true    # Set to true to save intermediate timestep results (matching launch.json)
HEIGHTS=(512)             # Panorama height (matching launch.json)
WIDTHS=(2048)             # Panorama width (matching launch.json)
HLAT=64                   # Latent height (matching launch.json)
WLAT=64                   # Latent width (matching launch.json)
FOV_DEG=80                # Field of view in degrees (matching launch.json)
OVERLAP=0.6               # Overlap between tiles (matching launch.json)
N_DIRS=75000              # Number of directions to sample (matching launch.json)
STEPS_LIST=(50)           # Inference steps (matching launch.json)
SEEDS=(0)                 # Seeds (matching launch.json)
GUIDANCE_SCALE=7.5        # Guidance scale (matching launch.json)

# 프롬프트 목록
PROMPTS=(
  "The city's skyline stretches below, clearly visible as vibrant fireworks light up the night sky. The fireworks burst in various colors, scattering across the air, while their reflections shimmer on the glass windows of skyscrapers. The camera smoothly pans across the city, capturing the river and bridges, with distant car lights creating a flowing effect."
  "An upward view from underwater, looking towards the surface where sunlight beams penetrate the clear ocean. Gentle ripples create a shimmering effect, and the water transitions from deep blue to a lighter, almost turquoise hue near the surface. The light refracts beautifully, creating a dreamlike underwater glow."
  "A stunning underwater scene filled with vibrant tropical fish swimming gracefully through the crystal-clear water. Various species, from small neon-colored fish to larger, elegant ones, move in harmony. The environment is serene, with the water gently flowing and reflecting the sunlight from above. The clarity of the water highlights the intricate details of the marine life."
  "A breathtaking top-down view of a colorful tropical coral reef. The ocean floor is covered with vibrant corals, ranging from bright orange and pink to deep purple and blue. Small fish dart between the coral formations, while gentle waves create a mesmerizing play of light and shadow on the seabed. The details of the marine ecosystem are incredibly vivid, showing the beauty of underwater biodiversity."
  "An upward view of the night sky. Soft moonlight filters through wispy clouds, casting a serene glow over the winter landscape."
  "From the snowy fields, a view toward a peaceful village nestled among snow-covered hills. Warm lights glow from the windows of small wooden cabins, contrasting with the crisp, cold air under the moonlit sky."
  "A high-angle view of snow-covered paths winding through the landscape. The fresh snow glistens under the moonlight, while the warm glow of lanterns and fireplaces reflects off the frosty roads, creating a cozy contrast against the cold night."
  "Dark clouds churned in slow, twisting spirals overhead, their shifting forms casting fleeting shadows below. The clouds thickened, their edges curling like ink dissolving in water, deepening into shades of charcoal and silver. Occasionally, bright patches pierced through the dense formations, creating stark contrasts of light and darkness in the sky."
  "Dark clouds churned in slow, twisting spirals above the rolling expanse of vibrant green grass, their shifting forms casting fleeting shadows across the land. A soft breeze rustled the blades, carrying the crisp scent of damp earth. The clouds thickened, their edges curling like ink dissolving in water, deepening in shades of charcoal and silver. Patches of sunlight briefly pierced through, creating shifting patterns of light and shadow that danced across the swaying field."
  "A rolling expanse of vibrant green grass stretched endlessly, each blade rustling softly in the gentle breeze. The crisp scent of damp earth lingered in the air as the grass swayed rhythmically, creating subtle waves across the field. The surface shimmered with varying shades of green, highlighting the texture and movement of the landscape."
  "Vibrant fireworks burst across the night sky, painting the heavens with shimmering trails of vivid colors. Some fireworks transform into heart shapes before fading, adding a touch of elegance to the display. The camera focuses on explosive arcs and sparkling embers, capturing every brilliant flash against an infinite, celestial canvas."
  "A breathtaking city skyline stretches below, illuminated by countless lights reflecting off towering skyscrapers. The camera smoothly pans across the landscape, revealing a river winding through the metropolis and bridges glowing under streetlights. Distant car headlights flow like streams of light, adding a dynamic rhythm to the urban nightscape."
  "Dark storm clouds swirl overhead as multiple lightning bolts strike at different moments, briefly illuminating the chaotic sky. The jagged bolts cut through the darkness, revealing shifting cloud formations in flashes of electric blue and white. The perspective is an upward view, emphasizing the storm’s immense scale. The scene is dynamic, with each lightning strike casting sharp contrasts of light and shadow across the turbulent sky."
  "A mid-angle view reveals the vast ocean meeting the storm-filled sky, where towering waves rise and fall beneath the relentless tempest. Lightning bolts crack through the heavy clouds, their electric glow reflecting off the turbulent water. The horizon is barely visible, obscured by mist and rain as gusting winds whip across the sea’s surface. Each flash of light briefly exposes the chaos, illuminating the swirling storm above and the restless waves below."
  "A high-angle view captures the vast, turbulent deep blue ocean as powerful waves crash and churn beneath the storm’s force. The water’s surface ripples with energy, each undulating motion reflecting the storm’s intensity. White foam swirls atop the restless sea, contrasting against the dark depths. Gusts of wind carve patterns into the waves, while distant flashes of lightning momentarily reveal the chaotic movement below."
  "An upward view of the vast night sky above an open field, with scattered fireflies drifting gently, their faint glows blending with distant stars."
  "A serene nighttime meadow, where countless fireflies flicker softly, casting a warm, golden light that dances over the swaying grass and wildflowers."
  "A dramatic top-down view of a vast open field grass, where waves of grass ripple in the night breeze, dotted with countless fireflies drifting just above, their soft glow flickering across the landscape."
)

NEGATIVE_PROMPTS=(
  "blurry, low resolution, distorted, noisy"
#   "low quality, blurry, deformed"
)

STAMP=$(date +"%Y%m%d_%H%M%S")
OUTDIR="outputs/test_${SD_VERSION}_gpu${GPU_NUM}_${STAMP}"
mkdir -p "$OUTDIR"

# 파일명에 쓸 slugify 함수 (공백/특수문자 제거)
slugify () {
  # shellcheck disable=SC2001
  echo "$1" \
    | sed -e 's/[^A-Za-z0-9._-]/_/g' \
    | sed -e 's/__*/_/g' \
    | sed -e 's/^_//' -e 's/_$//'
}

echo "[INFO] Running grid..."
echo "[INFO] Output dir: $OUTDIR"

n=0
for prompt in "${PROMPTS[@]}"; do
  pslug=$(slugify "$prompt")
  for neg in "${NEGATIVE_PROMPTS[@]}"; do
    nslug=$(slugify "${neg:-none}")
    for H in "${HEIGHTS[@]}"; do
      for W in "${WIDTHS[@]}"; do
        for steps in "${STEPS_LIST[@]}"; do
          for seed in "${SEEDS[@]}"; do
            n=$((n+1))
            ofn="${OUTDIR}/pan_${n}_${pslug}_neg-${nslug}_H${H}_W${W}_S${steps}_seed${seed}"
            echo "[RUN $n] H=$H W=$W Hlat=$HLAT Wlat=$WLAT FOV=$FOV_DEG overlap=$OVERLAP N_dirs=$N_DIRS steps=$steps seed=$seed GPU=$GPU_NUM guidance=$GUIDANCE_SCALE"
            echo "         prompt='${prompt}'"
            echo "         negative='${neg}'"
            
            # Build command with optional intermediate saving
            cmd="$PY $SCRIPT \
              --prompt \"$prompt\" \
              --negative \"$neg\" \
              --rf_version \"$SD_VERSION\" \
              --H \"$H\" \
              --W \"$W\" \
              --Hlat \"$HLAT\" \
              --Wlat \"$WLAT\" \
              --fov_deg \"$FOV_DEG\" \
              --overlap \"$OVERLAP\" \
              --N_dirs \"$N_DIRS\" \
              --steps \"$steps\" \
              --seed \"$seed\" \
              --gpu \"$GPU_NUM\" \
              --guidance_scale \"$GUIDANCE_SCALE\" \
              --outfolder \"$ofn\""
            
            # Add intermediate saving flag if enabled
            if [ "$SAVE_INTERMEDIATE" = true ]; then
              cmd="$cmd --save_intermediate"
              echo "         saving intermediate timesteps: ON"
              echo "         intermediate files will be saved in: $ofn/"
              echo "         tile files: tile_000_000.png, tile_001_000.png, ..."
            else
              echo "         saving only final result in: $ofn/"
            fi
            
            # Execute the command
            eval $cmd
          done
        done
      </dev/null
      done
    done
  done
done

echo "[DONE] Saved to: $OUTDIR"
