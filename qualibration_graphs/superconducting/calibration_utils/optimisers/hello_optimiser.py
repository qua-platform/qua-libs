# a 'check the random number' and return if it hits the target
# if it does not hit the target, resolve params with a new range
from qualibrate import QualibrationNode, QualibrationGraph
from typing import Any


def should_repeat_hello(node: QualibrationNode, target: str) -> bool:
    # by default will run again
    try:
        check_flag = node.results["success"]
    except:
        print("Not run yet")
        return True
    if check_flag == 0:
        return False  # found it, stop looping
    else:
        return True  # did not find it, keep looping


#
def hello_retry_params(node: QualibrationNode, target: str):
    try:
        check_flag = node.results["success"]

        l = node.results["guess_range"]["min"]
        r = node.results["guess_range"]["max"]
        prev = node.results["guess_range"]["guess"]

        if check_flag < 0:
            r = prev  # move upper bound to prev guess
        else:
            l = prev  # move lower bound to prev guess

        return {"min_n": l, "max_n": r}
    except:
        return


def hello_coinflip(node: QualibrationNode) -> bool:
    # check if the heads or tails results matches
    try:
        if node.results["coinflip"] == node.parameters.retry_on:
            return True
        else:
            return False
    except:
        return False  # exit anyway
