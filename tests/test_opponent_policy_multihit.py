import subprocess
from pathlib import Path


def test_multihit_ranking_uses_showdown_expectation() -> None:
    root = Path(__file__).resolve().parents[1]
    source = r"""
const assert = require("node:assert/strict");
const {createOpponentPolicyEngine} = require("./showdown/runtime/real_belief_probe/opponent_policy.cjs");
const attacker = {getStat(){return 100;}, getTypes(){return ["Normal"];}, hasAbility(id){return id === "skilllink" && this.linked;}, hasItem(){return false;}};
const defender = {getStat(){return 100;}};
const moves = new Map([
  ["multi", {id:"multi", exists:true, category:"Physical", basePower:30, accuracy:true, type:"Normal", multihit:[2,5]}],
  ["single", {id:"single", exists:true, category:"Physical", basePower:95, accuracy:true, type:"Normal"}],
]);
const battle = {
  restart(){}, destroy(){},
  p1:{active:[defender], sideConditions:{}},
  p2:{active:[attacker], pokemon:[attacker], activeRequest:{active:[{moves:[{id:"multi"},{id:"single"}]}]}},
  dex:{moves:{get(id){return moves.get(id);}}, getImmunity(){return true;}, getEffectiveness(){return 0;}},
};
const engine = createOpponentPolicyEngine({
  Battle:{fromJSON(){return battle;}}, benchFactorField:"opponent.bench.species", benchPrior:null,
  fail(message){throw new Error(message);}, instrumentPublicRootReads(){throw new Error("unexpected");},
  opponentPolicy:{kind:"strategy-mixture", strategies:[{kind:"max-damage"}], voluntary_switches:false},
  publicOpponentView(){return null;}, toID(value){return String(value).toLowerCase();},
});
const rank = () => engine.opponentDistributionForSnapshot({}).map(row => row.choice);
assert.deepEqual(rank(), ["move single"]);
attacker.linked = true;
assert.deepEqual(rank(), ["move multi"]);
"""
    result = subprocess.run(["node", "-e", source], cwd=root, check=True, capture_output=True, text=True)
    assert result.stderr == ""
