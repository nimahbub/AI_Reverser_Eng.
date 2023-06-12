
def generate_data(X, y, sequence_length = 1, step = 1):
    sequences = []
    targets = []
    for start in range(0, len(X) - sequence_length, step):
        end = start + sequence_length
        sequences.append(X[start:end])
        targets.append(y[end-1])
    
    sequences = np.array(sequences)
    targets = np.array(targets)

    # Step 3: Normalize or scale the input
    # Apply appropriate normalization or scaling techniques to sequences if required

    # Step 4: Split into training and test sets
    split_ratio = 0.6  # 80% for training, 20% for testing
    split_index = int(split_ratio * len(sequences))

    train_sequences = sequences[:split_index]
    train_targets = targets[:split_index]

    test_sequences = sequences[split_index:]
    test_targets = targets[split_index:]


    # Step 5: Create PyTorch data loaders
    train_dataset = TensorDataset(torch.Tensor(train_sequences), (torch.Tensor(train_targets)).type(torch.LongTensor) )
    test_dataset = TensorDataset(torch.Tensor(test_sequences), (torch.Tensor(test_targets)).type(torch.LongTensor))

    # Making DataLoader 
    train_loader = DataLoader(train_dataset,
                              batch_size=100,
                              shuffle=False
                              )

    test_loader = DataLoader(test_dataset,
                             batch_size=100,
                             shuffle=False)
    
    return train_loader, test_loader





def dataload (path):
    data = pd.read_csv(path)

    X = data.iloc[:, 0:5].values
    y = data.iloc[:, -1].values
    # Perform feature normalization
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    train_loader, test_loader = generate_data(X_scale,y)
    return  train_loader, test_loader



  