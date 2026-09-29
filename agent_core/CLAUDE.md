# target
implement agentic workflow for generating english news video
# tech stack
langgraph
langsmith
# workflow
1. read resource/text/source.txt, generate 1 min english news. split each sentence into 1 file in resource/text/
2. generate image prompt for each sentence in resource/image_prompt/
3. use tts model to generate audio files in resource/audio/
4. use zimage model to generate images in resource/images/
5. use ffmpeg to generate video, 1 image pairs 1 audio. Like step5.py
# spec
1. clear and straight implement
2. put tool in tools/ if necessary to write new tools


