import os
from pathlib import Path
import subprocess
import sys


def test_report_export_works_in_an_ascii_locale(tmp_path):
    code = r'''
import sys
from mms_eval.pipeline import write_report
report={'n':2,'protocol_id':'test\u4e2d', 'mms':{'value':.5,'status':'ok'},
        'distribution':{k:{'value':0.,'status':'ok','method':'MMD\u00b2'} for k in ('fid','kid','is','precision_recall')}}
write_report(sys.argv[1],report)
'''
    env = {'PATH': os.defpath, 'LC_ALL':'C', 'LANG':'C', 'PYTHONUTF8':'0', 'PYTHONCOERCECLOCALE':'0'}
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path/'report.json')], env=env,
                            cwd=Path(__file__).resolve().parents[1], capture_output=True)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')
    assert 'test中' in (tmp_path/'report.html').read_text(encoding='utf-8')
