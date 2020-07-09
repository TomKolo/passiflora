#!/bin/bash

mkdir raw
mkdir divided

fileid=("164X1yU6x7rwaK1_NRXhfoT24TWEhMCso" "1MiEoxb5mpt8LZjxMZ6YvCgkB-82QX2T8" "10MTy1ltLn6c8TaZUPPhkmIISx6LSyXq3" \
"1vFwCZu8lEZ3E7Ky9q1B_xmd21qyKkggL" "1zdMMbNKl6TzawuQJyzudW22qoX5YzkDk" "1RSs7pThaBFpvjwz-IRy0W2OK6z9iYDHP" "1Diqw5JyHXwlHMeb0DYfrYDDL_o_waOAU" "1z9WDyc1gfUdn3YNHxY0LhUtlk1_2MUUN" \
"1w3kaSMmrv20DCPA_OkjSc4ej1OLmEFCd" "1ih9ErOxRe9KqwcdjHcl_HFxDdpUtrvdt" "1o2yCP6l3YnAS9UFNFBYu-FkLjifg16ME" "1RZG5bA-g0t6F4hI-o9qkey-RgahWGHCe" "1uddhh2vig7TsDQ2iSYd9h0q6GKFJk8cV" \
"1vpA8aZYF_j8kc9VhSnZ3JfE_o3cAGfmN" "1QH8eagPc3PdTONhFDYZk-u5gq53xhKyY" "1lhOq8vWjhjl8BqnAqaX2XcbW4yC8oVrB" "1iWYibY2_yW2JR33twnTm8msGp-gWL1wg" "1gAFP7PVARh2zJrrQKRg3JPLWjiVub6Vs" \
"1g1u2__28bmwnayJy1Np6yZmxNxGB6_LD" "1N3xHcX4aleTv9oneSeQvDpNLVeeEhXM4" "1U42tX5f7RnsjSeEDc7xR0I72n1xAjhh4" "1rACtAkXFQrq1kwBvCf86BA6MtRqrRZbG" "1sDIHtqe8LENjDXBczFGnjvoKbNEoxDJb" \
"1SYC3i2Y-n0OMCvxQIYaYNp9guUqZYhTG" "1qU5PZjZhAxedrvCHktkhlpb1ycGrDF_8" "1qCnBbEfPW-L4gaBlFLPS-7mUVqhJYgGT" "1fwvwBPcyUP7C7XBFLSleBjFCgOv8O3v7" "1HfbjCLhRaWhzhZz5LpMlXliuBjsovGD9" \
"1mal_5ytSkEvpOlfPsaKOncghhdB6LT8R" "1cAmXVFW2q9I9NKJg9cNKllbUfG0jAMh_" "1kiruC1VfALrmBY77WI29MhAPsPZFdTmw" "1cp5KIXgToPkhHnMAL0ZW1yLN8_wUflv8" "1MvGTuO-vDNYR43ihQwcgQ7tMYw6KYgls" \
"1HL5zEyp3VP_Wh2WWuPPYeOGfloh3EwIk" "1lEYzUPaZfdNRRSUqUtBXAdNluQOn1MRm")
for x in {0..34}
do
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid[$x]}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid[$x]}" -o raw/all_data_$x.json
done

rm cookie