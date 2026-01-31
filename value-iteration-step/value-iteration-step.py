def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    numStates = len(values)
    newValues = [0.0] * numStates

    for s in range(numStates):
        actionValues = []
        for a in range(len(transitions[s])):
            expectedFutureValue = 0
            for sPrime in range(numStates):
                expectedFutureValue += transitions[s][a][sPrime] * values[sPrime]
            qValue = rewards[s][a] + gamma * expectedFutureValue
            actionValues.append(qValue)
        newValues[s] = max(actionValues)
    return newValues