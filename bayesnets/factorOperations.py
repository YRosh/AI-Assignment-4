from bayesNet import Factor
import operator as op
import util
import functools

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors, joinVariable):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


def joinFactors(factors):
    """
    Question 3: Your join implementation 

    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))


    "*** YOUR CODE HERE ***"
    uncondVars = set() # list of all unique unconditioned variables
    condVars = set() # list of all unique conditioned variables

    # Iterating over all factors
    for factor in factors:
        # Adding all unconditioned variables of this factor to a list
        for var in factor.unconditionedVariables():
            uncondVars.add(var)
        # Adding all conditioned variables of this factor to separate list
        for var in factor.conditionedVariables():
            condVars.add(var)

    # Removing any unconditioned variables if present in the conditioned variables list
    for var in uncondVars:
        if var in condVars:
            condVars.remove(var)

    # Creating new factor from set of all conditioned and unconditioned variables
    newFac = Factor(uncondVars, condVars, list(factors)[0].variableDomainsDict())

    # Iterating over all the possible assignments for the new joined factor
    for assign in newFac.getAllPossibleAssignmentDicts():
        # calculating overall joint probability for this assignment over all factors
        prod = 1
        for factor in factors:
            prod *= Factor.getProbability(factor, assign) # probability of this assignment in current factor
        # setting probability of this assignment for the new joined factor
        Factor.setProbability(newFac, assign, prod)
    # returning new joined factor with probabilities of possible assignments
    return newFac
    "*** END YOUR CODE HERE ***"

def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor, eliminationVariable):
        """
        Question 4: Your eliminate implementation 

        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        # list of all unconditioned variables for the given factor
        uncondVars = factor.unconditionedVariables()
        # removing the eliminationVariable from this list
        uncondVars.remove(eliminationVariable)

        # creating new factor with updated unconditioned variables and existing conditioned variables
        newFac = Factor(uncondVars, factor.conditionedVariables(), factor.variableDomainsDict())
        # Iterating over all the assignments for the new factor
        for assign in newFac.getAllPossibleAssignmentDicts():
            s = 0 # overall probability for this assignment
            # iterating over domain of the eliminated variable for this factor
            for var in factor.variableDomainsDict()[eliminationVariable]:
                assign[eliminationVariable] = var
                # Adding all the probabilities of the assign with each assignment of the eliminated variable
                s += factor.getProbability(assign)
            # setting the new probability of the assign without the eliminationVariable
            newFac.setProbability(assign, s)

        # returning new factor after removing the variable and updating its probability
        return newFac
        "*** END YOUR CODE HERE ***"

    return eliminate

eliminate = eliminateWithCallTracking()


def normalize(factor):
    """
    Question 5: Your normalize implementation 

    Input factor is a single factor.

    The set of conditioned variables for the normalized factor consists 
    of the input factor's conditioned variables as well as any of the 
    input factor's unconditioned variables with exactly one entry in their 
    domain.  Since there is only one entry in that variable's domain, we 
    can either assume it was assigned as evidence to have only one variable 
    in its domain, or it only had one entry in its domain to begin with.
    This blurs the distinction between evidence assignments and variables 
    with single value domains, but that is alright since we have to assign 
    variables that only have one value in their domain to that single value.

    Return a new factor where the sum of the all the probabilities in the table is 1.
    This should be a new factor, not a modification of this factor in place.

    If the sum of probabilities in the input factor is 0,
    you should return None.

    This is intended to be used at the end of a probabilistic inference query.
    Because of this, all variables that have more than one element in their 
    domain are assumed to be unconditioned.
    There are more general implementations of normalize, but we will only 
    implement this version.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    variableDomainsDict = factor.variableDomainsDict()
    for conditionedVariable in factor.conditionedVariables():
        if len(variableDomainsDict[conditionedVariable]) > 1:
            print("Factor failed normalize typecheck: ", factor)
            raise ValueError("The factor to be normalized must have only one " + \
                            "assignment of the \n" + "conditional variables, " + \
                            "so that total probability will sum to 1\n" + 
                            str(factor))

    "*** YOUR CODE HERE ***"
    # Calculating the overall probability for all assignments for the factor
    probS = 0
    for assign in factor.getAllPossibleAssignmentDicts():
        probS += factor.getProbability(assign)

    uncondVars = set() # list of all unconditioned variables in the factor
    condVars = set() # list of all conditioned variables in the factor

    # Adding conditioned variables to the list
    for var in factor.conditionedVariables():
        condVars.add(var)

    # Iterating over unconditioned variables in the factor
    for var in factor.unconditionedVariables():
        # if variable has one value in domain adding it to conditioned variables list
        # else to unconditioned variables list
        if len(factor.variableDomainsDict()[var]) == 1:
            condVars.add(var)
        else:
            uncondVars.add(var)

    # creating new normalized factor with the conditioned and unconditioned variables list
    newFac = Factor(uncondVars, condVars, factor.variableDomainsDict())

    # iterating over all the possible assignments of the factor
    for assign in newFac.getAllPossibleAssignmentDicts():
        # calculating the new normalized probability of this assignment
        newProb = factor.getProbability(assign) / probS
        # setting this new probability to this assignment
        newFac.setProbability(assign, newProb)

    # returning new normalized factor with updated probability
    return newFac
    "*** END YOUR CODE HERE ***"

