# fixed_scaling picked a0p000 = the IMAGE-ONLY endpoint, where the tabular branch is scaled
# to zero, so tabular perturbation is a no-op and every mode is exactly 0. That is a useful
# sanity check but not a representative family value -- add the mid-alpha setting.
import os, runpy, sys
os.environ["SIGNED_SETTING"]="a0p500"
exec(open(os.path.join(os.path.dirname(__file__),"run_signed.py")).read().replace(
 'picks=master[has].groupby("impl_name").head(int(os.environ.get("PER_FAMILY","1")))',
 'picks=master[has & master.setting_tag.eq("a0p500") & master.seed.eq(47)]'))
