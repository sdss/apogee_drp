from apogee_drp.apred.cal import makecal as makecal_module
from apogee_drp.apred.cal.dependencies import CalibrationGraph, CalibrationNode


class FakeLoad:
    apred = 'test'
    telescope = 'apo25m'

    def exists(self, root, num=None):
        return False


def context():
    return makecal_module.CalibrationContext(
        load=FakeLoad(), calfile='unused.par', allcaldict={}
    )


def test_run_calibration_graph_builds_prerequisites_first(monkeypatch):
    detector = CalibrationNode('detector', '1')
    dark = CalibrationNode('dark', '2')
    flat = CalibrationNode('flat', '3')
    graph = CalibrationGraph(
        roots={flat},
        dependencies={detector: set(), dark: {detector}, flat: {dark}},
    )

    class FakeResolver:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self, roots):
            assert roots == [('flat', '3')]
            return graph

    monkeypatch.setattr(makecal_module, 'CalibrationDependencyResolver',
                        FakeResolver)
    calls = []
    builders = {
        'det': makecal_module.BuilderSpec(
            'det', 'Detector', lambda name, ctx: calls.append(('det', name))
        ),
        'dark': makecal_module.BuilderSpec(
            'dark', 'Dark', lambda name, ctx: calls.append(('dark', name))
        ),
        'flat': makecal_module.BuilderSpec(
            'flat', 'Flat', lambda name, ctx: calls.append(('flat', name))
        ),
    }
    monkeypatch.setattr(makecal_module, 'BUILDERS', builders)

    makecal_module._run_calibration_graph('3', 'flat', context())

    assert calls == [('det', '1'), ('dark', '2'), ('flat', '3')]


def test_run_calibration_graph_skips_existing_nodes(monkeypatch):
    dark = CalibrationNode('dark', '2')
    flat = CalibrationNode('flat', '3')
    graph = CalibrationGraph(
        roots={flat},
        dependencies={dark: set(), flat: {dark}},
        existing={dark},
    )

    class FakeResolver:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self, roots):
            return graph

    monkeypatch.setattr(makecal_module, 'CalibrationDependencyResolver',
                        FakeResolver)
    calls = []
    builders = {
        'dark': makecal_module.BuilderSpec(
            'dark', 'Dark', lambda name, ctx: calls.append(('dark', name))
        ),
        'flat': makecal_module.BuilderSpec(
            'flat', 'Flat', lambda name, ctx: calls.append(('flat', name))
        ),
    }
    monkeypatch.setattr(makecal_module, 'BUILDERS', builders)

    makecal_module._run_calibration_graph('3', 'flat', context())

    assert calls == [('flat', '3')]


def test_every_dependency_type_has_a_registered_builder():
    expected = {
        'det', 'dark', 'flat', 'bpm', 'fiber', 'sparse', 'littrow',
        'psf', 'modelpsf', 'fpi', 'persist', 'persistmodel', 'flux',
        'response', 'wave', 'multiwave', 'dailywave', 'telluric', 'lsf',
    }

    assert set(makecal_module.BUILDERS) == expected


def test_context_preserves_unknown_builder_options():
    ctx = makecal_module._context_from_arguments(
        None,
        None,
        {
            'load': FakeLoad(),
            'calfile': 'unused.par',
            'allcaldict': {},
            'clobber': True,
            'custom_option': 42,
        },
    )

    assert ctx.clobber is True
    assert ctx.option('custom_option') == 42
