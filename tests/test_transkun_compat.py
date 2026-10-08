from audio2score.compat import _replace_audio_loader, _replace_pkg_resources


def test_patch_published_transkun_source_fragments():
    source = '''import argparse\n\ndef readAudio(path,  normalize= True):\n    import pydub\n    audio = pydub.AudioSegment.from_mp3(path)\n    y = np.array(audio.get_array_of_samples())\n    y = y.reshape(-1, audio.channels)\n    if normalize:\n        y =  np.float32(y)/2**15\n    return audio.frame_rate, y\n\ndef main():\n    import pkg_resources\n    defaultWeight =  (pkg_resources.resource_filename(__name__, "pretrained/2.0.pt"))\n    defaultConf =  (pkg_resources.resource_filename(__name__, "pretrained/2.0.conf"))\n'''
    source, pkg = _replace_pkg_resources(source)
    source, audio = _replace_audio_loader(source)
    assert pkg is True
    assert audio is True
    assert "pkg_resources" not in source
    assert "import pydub" not in source
    assert "AudioSegment.from_mp3" not in source
    assert "soundfile as sf" in source
    assert "Path(__file__).resolve().parent" in source


def test_patches_are_noops_after_application():
    source = '''def readAudio(path, normalize=True):\n    import soundfile as sf\n    y, fs = sf.read(path, dtype="float32", always_2d=True)\n    return fs, y\n'''
    new, changed = _replace_audio_loader(source)
    assert new == source
    assert changed is False
