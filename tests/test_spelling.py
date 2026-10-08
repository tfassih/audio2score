from audio2score.music import midi_to_pitch_components_for_key, pc_name_for_key, chord_name


def test_d_sharp_minor_spells_e_sharp_not_f():
    # D# minor has six sharps; MIDI 65 should notate as E#4.
    assert midi_to_pitch_components_for_key(65, 6) == ('E', 1, 4)
    assert pc_name_for_key(5, 6) == 'E#'
    assert chord_name(1, 'maj', 5, fifths=6) == 'C#/E#'
