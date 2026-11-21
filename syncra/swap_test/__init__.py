"""The swap test: does an answer follow the audio-visual pairing present in the input?

Each source clip has four spoken cues in four equal-length windows. Exchanging
windows 1<->3 and 2<->4 in the audio stream, the video stream, or both gives
four input versions per cue; a cue is credited only if all four are answered
correctly (AllFour).
"""
