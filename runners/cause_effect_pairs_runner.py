from CauseEffectPairs.Run_CEP_01 import runCEPair


def RunCauseEffectPairs():
    # define some simulation parameters
    skipPairs = [52, 54, 55]
    # skip these pairs, as indicated by Mooij et al (2016) because the variables
    # are not bivariate (i.e., X and Y are not univariate)

    SplitPerc = .8
    scaleDat = True
    priorDist = 'laplace'
    epochs = 500 + 250
    optMethod = 'schedule'
    correctCount = 0
    runningCount = 0
    BinaryCutoff = .15
    correctCount_nobinary = 0
    runningCount_nobinary = 0
    LayerList = [1, 3]
    depthList = [5]

    for i in range(1, 108):
        if i in skipPairs:
            pass
        else:
            res, predModel, trueModel, ctsRatio = runCEPair(i, LayerList, depthList, priorDist=priorDist,
                                                            TrainSplit=SplitPerc, epochs=epochs, optMethod=optMethod,
                                                            scaleDat=scaleDat, verbose=True)

            runningCount += 1
            if ctsRatio > BinaryCutoff:
                runningCount_nobinary += 1
            if predModel.replace('-', '') == trueModel.replace('-', ''):
                print('Correct!')
                correctCount += 1
                if ctsRatio > BinaryCutoff:
                    correctCount_nobinary += 1

            print('running mean: ' + str(float(correctCount) / runningCount))
