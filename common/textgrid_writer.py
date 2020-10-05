import textgrid

def __separate_intervals(y, inoise):
    breakpoints = __get_breakpoints(inoise)
    is_signal = True
    start_point = 0
    for bp in breakpoints:
        yield is_signal, start_point, bp
        start_point = bp
        is_signal = not is_signal
    yield is_signal, start_point, len(y)


def __get_breakpoints(inoise):
    expected = inoise[0] + 1
    yield inoise[0]
    for i in inoise[1:]:
        if i != expected:
            yield expected
            yield i
        expected = i + 1


def audio_to_textgrid(y, sr, inoise) -> textgrid.TextGrid:
    '''
        Converts a piece of audio into a praat's textgrid format.
    '''
    max_time = len(y) / sr
    tg = textgrid.TextGrid(maxTime=max_time)
    tier = textgrid.IntervalTier(name='silêncio', maxTime=max_time)

    intervals = __separate_intervals(y, inoise)
    for is_signal, imin, imax in intervals:
        time_min, time_max = imin / sr, imax / sr
        mark = 'locucao' if is_signal else 'pausa'
        tier.addInterval(textgrid.Interval(minTime=time_min, maxTime=time_max, mark=mark))

    tg.append(tier)
    return tg


def write_textgrid_to_file(filename: str, audio_filename: str, tg: textgrid.TextGrid):
    ''' write a textgrid to a file. '''
    tg.name = audio_filename
    with open(filename, 'w') as f:
        tg.write(f)