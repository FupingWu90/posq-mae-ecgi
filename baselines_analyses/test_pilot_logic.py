"""Offline test of the queue logic in training_sweeps/pilots.py with simulated results (no file of the real queue
is touched: decision files go to a temporary folder, results are mocked)."""
import os, sys, json, tempfile, shutil
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'training_sweeps'))
import pilots as P

REF = P.v1_ref()


def run(scn, expect_seq, expect_base=None, expect_heads=None, converging=False):
    tmp = tempfile.mkdtemp(); P.STATE = tmp; P.FAILED = os.path.join(tmp, 'failed'); os.makedirs(P.FAILED)
    done = {}

    def fake_conv():
        j = P._load(P._state('base_decision.json'))
        if not j or j['base'] not in done:
            return None
        return dict(still=converging, best_epoch=799, gain_last_200=0.003, gain_second_half=0.01, base_val_mae=done[j['base']][0])
    P.base_still_converging = fake_conv

    def fake_result(pid, base):
        if P.PILOTS[pid][1] and base is None:
            return None
        if pid not in done:
            return None
        cmd = P.args_of(pid, base)
        return dict(id=pid, run_dir='x', extra=cmd[len(P.COMMON):-2], val_mae=done[pid][0], val_AT=done[pid][1])
    P.result = fake_result
    seq, cmds = [], {}
    for _ in range(30):
        nxt = P.next_pilot()
        seq.append(nxt)
        if nxt.startswith(('DONE', 'STOP')):
            break
        cmds[nxt] = ' '.join(P.args_of(nxt, P.flags_for(nxt))[len(P.COMMON):])
        if scn.get(nxt) == 'fail':
            open(os.path.join(P.FAILED, nxt), 'w').close()
        else:
            done[nxt] = scn.get(nxt, (REF - 0.001, 13.0))
    hj, bj = P._load(P._state('heads_decision.json')), P._load(P._state('base_decision.json'))
    ok = [s.split(':')[0] for s in seq] == expect_seq and (expect_base is None or (bj or {}).get('flags') == expect_base) \
        and (expect_heads is None or (hj or {}).get('flags') == expect_heads)
    shutil.rmtree(tmp)
    return ok, seq, cmds, hj, bj


TAIL = ['LOC3', 'LOC2', 'LOC1', 'MSK1', 'MSK3', 'LOC3B', 'LOC4', 'EMB1', 'LOC3D', 'LOC5', 'DEC4', 'EPL3', 'CAPE', 'CAPD', 'CAPW', 'DONE']
TAIL_EPO = ['LOC3', 'LOC2', 'LOC1', 'MSK1', 'MSK3', 'LOC3B', 'LOC4', 'EMB1', 'LOC3D', 'LOC5', 'DEC4', 'EPL3', 'EPO', 'CAPE', 'CAPD', 'CAPW', 'DONE']
tests = [
    ('heads tie, tsa tie -> base DEC1', {'DEC1': (REF - 0.003, 13.0), 'DECH': (REF - 0.004, 13.0), 'DEC2': (REF - 0.004, 12.9)},
     ['DEC1', 'DECH', 'DEC2'] + TAIL, [], []),
    ('4 heads better, tsa tie -> base = 4 heads', {'DEC1': (REF - 0.003, 13.0), 'DECH': (REF - 0.008, 12.6), 'DEC2': (REF - 0.009, 12.5)},
     ['DEC1', 'DECH', 'DEC2'] + TAIL, ['--dec_heads', '4'], ['--dec_heads', '4']),
    ('4 heads better, tsa better -> base = 4 heads + tsa', {'DEC1': (REF - 0.003, 13.0), 'DECH': (REF - 0.008, 12.6), 'DEC2': (REF - 0.012, 12.0)},
     ['DEC1', 'DECH', 'DEC2'] + TAIL, ['--dec_heads', '4', '--dec_tsa'], ['--dec_heads', '4']),
    ('heads worse, tsa wins on downstream AT at tied MAE', {'DEC1': (REF - 0.003, 13.0), 'DECH': (REF + 0.001, 13.2), 'DEC2': (REF - 0.004, 12.3)},
     ['DEC1', 'DECH', 'DEC2'] + TAIL, ['--dec_tsa'], []),
    ('DECH fails twice -> 16 heads kept', {'DEC1': (REF - 0.003, 13.0), 'DECH': 'fail', 'DEC2': (REF - 0.003, 13.0)},
     ['DEC1', 'DECH', 'DEC2'] + TAIL, [], []),
    ('all trail v1 -> DEC3 rescues', {'DEC1': (REF + 0.006, 13.5), 'DECH': (REF + 0.007, 13.5), 'DEC2': (REF + 0.005, 13.4), 'DEC3': (REF - 0.001, 13.0)},
     ['DEC1', 'DECH', 'DEC2', 'DEC3'] + TAIL, ['--dec_depth', '6'], []),
    ('all trail v1, DEC3 too -> STOP', {'DEC1': (REF + 0.006, 13.5), 'DECH': (REF + 0.007, 13.5), 'DEC2': (REF + 0.005, 13.4), 'DEC3': (REF + 0.006, 13.4)},
     ['DEC1', 'DECH', 'DEC2', 'DEC3', 'STOP'], None, []),
    ('a single-factor pilot fails twice -> skipped', {'LOC2': 'fail'},
     ['DEC1', 'DECH', 'DEC2', 'LOC3', 'LOC2', 'LOC1', 'MSK1', 'MSK3', 'LOC3B', 'LOC4', 'EMB1', 'LOC3D', 'LOC5', 'DEC4', 'EPL3', 'CAPE', 'CAPD', 'CAPW', 'DONE'], [], []),
]
tests_conv = [
    ('still converging, 1600 epochs help -> capacity checks at 1600', {'DEC1': (REF - 0.003, 13.0), 'DECH': (REF - 0.004, 13.0), 'DEC2': (REF - 0.004, 12.9), 'EPO': (REF - 0.008, 12.5)},
     ['DEC1', 'DECH', 'DEC2'] + TAIL_EPO, [], []),
    ('still converging, 1600 epochs do not help -> capacity checks at 800', {'DEC1': (REF - 0.003, 13.0), 'DECH': (REF - 0.004, 13.0), 'DEC2': (REF - 0.004, 12.9), 'EPO': (REF - 0.004, 13.0)},
     ['DEC1', 'DECH', 'DEC2'] + TAIL_EPO, [], []),
    ('still converging, EPO fails twice -> skipped, 800 kept', {'EPO': 'fail'},
     ['DEC1', 'DECH', 'DEC2'] + TAIL_EPO, [], []),
    ('4 heads + tsa base, still converging, 1600 help', {'DEC1': (REF - 0.003, 13.0), 'DECH': (REF - 0.008, 12.6), 'DEC2': (REF - 0.012, 12.0), 'EPO': (REF - 0.016, 11.8)},
     ['DEC1', 'DECH', 'DEC2'] + TAIL_EPO, ['--dec_heads', '4', '--dec_tsa'], ['--dec_heads', '4']),
]
allok = True
for name, scn, seq, base, heads in tests_conv:
    ok, got, cmds, hj, bj = run(scn, seq, base, heads, converging=True)
    allok &= ok
    print(('PASS' if ok else 'FAIL'), '|', name, '| EPO:', cmds.get('EPO'), '| CAPD:', cmds.get('CAPD'), '| CAPE:', cmds.get('CAPE'))
    if not ok:
        print('    got sequence:', got)
for name, scn, seq, base, heads in tests:
    ok, got, cmds, hj, bj = run(scn, seq, base, heads)
    allok &= ok
    print(('PASS' if ok else 'FAIL'), '|', name, '| base flags:', (bj or {}).get('flags'), '| DEC2:', cmds.get('DEC2'), '| CAPD:', cmds.get('CAPD'), '| CAPW:', cmds.get('CAPW'))
    if not ok:
        print('    got sequence:', got)
print('RESULT:', 'PASS' if allok else 'FAIL')
