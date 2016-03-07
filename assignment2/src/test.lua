function splitDataset(d, l, ratio)
   local shuffle = torch.randperm(data:size(1))
   local numTrain = math.floor(shuffle:size(1) * ratio)
   local numTest = shuffle:size(1) - numTrain

   local trainData = torch.ByteTensor(numTrain, d:size(2), d:size(3), d:size(4))
   local testData = torch.ByteTensor(numTest, d:size(2), d:size(3), d:size(4))
   local trainLabels = torch.ByteTensor(numTrain)
   local testLabels = torch.ByteTensor(numTest)

   for i=1,numTrain do
      trainData[i]:copy(d[shuffle[i]])
      trainLabels[i] = l[shuffle[i]]
   end
   for i=numTrain+1,numTrain+numTest do
      testData[i-numTrain]:copy(d[shuffle[i]])
      testLabels[i-numTrain] = l[shuffle[i]]
   end
   return trainData, trainLabels, testData, testLabels
end
