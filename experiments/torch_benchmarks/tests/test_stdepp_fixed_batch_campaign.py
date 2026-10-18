import json
import pytest
import stdepp_fixed_batch_campaign as runner
from first_group import make_model_and_points


def test_exact_original_targets_and_added_candidates_with_fixed_batch(tmp_path):
    plan = runner.plan(tmp_path)
    targets = [r['target'] for r in plan['cells']]
    assert targets == list(runner.TARGETS)+['111122223333','111112222233333']
    assert len(targets)==len(set(targets))==8
    assert all((r['batch'],r['width'],r['depth'])==(1000,128,4) for r in plan['cells'])
    assert all(r['seeds']==[20260908,20260909,20260910] for r in plan['cells'])
    assert all((r['warmups'],r['repeats'])==(10,30) for r in plan['cells'])
    assert plan['fresh_rerun'] and not plan['checkpoints']


def test_all_eight_targets_have_independent_analytic_validation():
    rows = runner.core.independent_checks('cpu', runner.EXPONENTS)
    assert len(rows)==16 and all(r['passed'] for r in rows)


@pytest.mark.parametrize('exponents,counts',list(zip(runner.EXPONENTS,[(1,1),(3,6),(5,13),(32,32),(14,40),(10,28),(13,49),(20,100)])))
def test_exact_common_backend_direction_counts(exponents,counts):
    model,points=make_model_and_points(20260908,2,2,2,'cpu')
    _,metadata,_=runner.core.build_evaluators(model,points,runner.core.alpha_from_exponents(exponents))
    assert tuple(metadata[m]['backend']['curves'] for m in runner.core.METHODS)==counts
    assert all(r['common_backend']=='torch_pinn.sparse_jets' for r in metadata.values())


def test_fixed_batch_receipt_cannot_reuse_old_scaling_gate(tmp_path):
    folder=tmp_path/'preflight';folder.mkdir();(folder/'status.txt').write_text('COMPLETE\n')
    record=dict(status='PASS',protocol=runner.core.PROTOCOL,source_fingerprint=runner.core.source_fingerprint(),environment={'gpu':'CPU_LOCAL_VALIDATION_ONLY'},batch=1000,width=128)
    (folder/'PASS.json').write_text(json.dumps(record))
    with pytest.raises(RuntimeError,match='mismatch'):runner.receipt(tmp_path,'cpu')
    record['protocol']=runner.PROTOCOL
    (folder/'PASS.json').write_text(json.dumps(record))
    assert runner.receipt(tmp_path,'cpu')['batch']==1000
