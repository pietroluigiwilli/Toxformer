import torch
import numpy as np
import pandas as pd
import rdkit
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import QED
from rdkit.Chem import MolSurf
from rdkit.Chem import rdFreeSASA as SASA


def properties(smile):
    ### calculates the QED property vector from smiles ###
    mol = Chem.MolFromSmiles(str(smile)) #makes an RDKIT molecule object from the smile string
    SA = MolSurf.LabuteASA(mol) #Molecular surface area
    props = list(QED.properties(mol)[:])
    props.append(SA)
    return props

def star(df,col,test=None):
    ### divides the LD50_rat values into 3 categories based on percentiles ###
    df["index"]=list(range(len(df)))
    df = df.set_index("index") #resets index
    first = df[col].quantile(1/3)
    onestar = df.query(f"{col}<={first}")
    df.loc[onestar.index, f"{col}_level"] = 1
    print(f"first percentile (1/3): {first}")
    second = df[col].quantile(2/3)
    twostar = df.query(f"{col}<={second} and {col}>{first}")
    df.loc[twostar.index, f"{col}_level"] = 2
    print(f"second percentile (2/3): {second}")
    threestar = df.query(f"{col}>{second}")
    df.loc[threestar.index, f"{col}_level"] = 3

    plt.figure()
    plt.xlim(0,10000)
    sb.histplot(data=df,x=col, hue=f"{col}_level",palette="viridis",binwidth=100)
    plt.savefig("data_level_distribution.pdf")
    plt.show()

    plt.figure()
    plt.xlim(0,10000)
    sb.histplot(data=df,y=f"{col}_level",x=col)
    plt.savefig("level_distribution.pdf")
    plt.show()

    if type(test)==pd.core.frame.DataFrame: # this is necessary if we want to split the testset with the same percentiles as the trianing set 
        # We have to extract the percentiles of the training set first.
        test["index"]=list(range(len(test)))
        test = test.set_index("index")
        onestar = test.query(f"{col}<={first}")
        test.loc[onestar.index, f"{col}_level"] = 1
        twostar = test.query(f"{col}<={second} and {col}>{first}")
        test.loc[twostar.index, f"{col}_level"] = 2
        threestar = test.query(f"{col}>{second}")
        test.loc[threestar.index, f"{col}_level"] = 3
        return test

    else: 
        return df
    
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, purpose, shuffle, representation="token", qed=True, tensor=True, max_len=None):
        super().__init__()
        self.data = data
        self.qed = qed 
        self.label = label
        print("\n"+purpose.title())
        
        # splitting the Dataframe
        if purpose == "train":
            self.data = self.data[:int(len(self.data)*0.7)]
        elif purpose == "all":
            self.data = self.data
        elif purpose == "val":
            self.data = self.data[int(len(self.data)*0.7):]
        elif purpose == "test":
            try:
                self.data = pd.read_csv("DATA/test_df.csv")
            except:
                self.data = pd.read_csv("/content/drive/My Drive/Toxicity Project/DATA/test_df.csv")
        else:
            print("purpose must be one of: train, all, val, test.")
        
        # Shuffling the Dataframe
        if shuffle:
            self.data = self.data.sample(frac=1, random_state=42)
        
        # Re-indexing is useful both in case the shuffling has been performed but also in case some smples have been dropped. 
        self.data["index"] = list(range(len(self.data)))
        self.data = self.data.set_index("index")

        if representation=="token":
            # use tokenized smiles
            self.data = self.data_tokens(max_len)
        elif representation=="finger":
            # use Morgan Fingerprints Bit Vectors
            self.data = self.data_fingers(bits=128)      
        elif representation=="both":
            # use both
            self.data = self.data_fingertokens(bits=128,max_len=max_len)
        
        self.max_map = np.max(self.data[:,1:].flatten())
        self.n_map = len(np.unique(self.data[:,1:].flatten()))
        print(f"Largest value: {self.max_map}")
        print(f"Number of unique embedding values: {self.n_map}")
            
        self.x = self.data[:,1:] 
        self.t = self.data[:,0] 
        
        if "level" in self.label:
            self.t -= 1
        
        if tensor==True:
            # convert to tensor
            self.x = torch.tensor(self.x)
            self.t = torch.tensor(self.t)
            if "level" in self.label:
                # One-Hot Encoding
                self.t = torch.eye(int(max(self.t).item())+1)[self.t.long()]

        print(f"x shape: {self.x.shape}")
        print(f"t shape: {self.t.shape}")
       
    def data_fingertokens(self, bits, max_len=None):
        # make both the tokenized smiles and the fingerprint vectors
        fingers = []
        ignore = 0
        if max_len == None:
            self.max_len = self.data.moldb_smiles.str.len().max()+4
        else:
            self.max_len = max_len
        print(f"Length of longest string: {self.max_len}")
        for i, smile in enumerate(self.data.moldb_smiles):
            try:
                tok = smile
                start = "%"
                stop = "|"
                pad = "_"
                if start in tok:
                    print("unvaiable start token")
                if stop in tok:
                    print("unvaiable stop token")
                if pad in tok:
                    print("unvaiable pad token")
                tok = start+" "+tok+" "+stop
                tok = tok+pad*(self.max_len-len(tok)+1) #pad the tokens
                token = np.array([ord(char) for char in list(tok)]) # encode the strings into integer unicode value
                finger = Chem.MolFromSmiles(smile)
                finger = np.asarray(Chem.GetMorganFingerprintAsBitVect(finger,2,bits)) # get fingerprint vector
                if self.qed==True:
                    props = self.properties(smile) # Calculate the QED property vector 
                    finger = np.append(finger,props)
                    
                fingertok = np.append(token,finger)
                ld = (float(self.data.loc[i,self.label]))
                ld = np.array(ld) # target values are appended to the finger in order to preserve the ordering in case one molecule is ommitted due to a typeerror
                fingers.append(np.append(ld,fingertok))

            except TypeError: # necessary to ignore some of the molecules as they are chemically impossible 
                ignore+=1
                continue
        print(f"ignored: {ignore}")
        print(f"token lengths: {len(token)}")
        print(f"finger lengths: {len(finger)}")
        return np.stack(fingers)

    def data_fingers(self, bits):
        # make the fingerprint vectors 
        fingers = []
        ignore = 0
        for i, smile in enumerate(self.data.moldb_smiles):
            try:
                finger = Chem.MolFromSmiles(smile)
                finger = np.asarray(Chem.GetMorganFingerprintAsBitVect(finger,2,bits))
                if self.qed==True:
                    props = self.properties(smile)
                    finger = np.append(finger,props)
                ld = (float(self.data.loc[i,self.label]))
                ld = np.array(ld)
                fingers.append(np.append(ld,finger))

            except TypeError: #necessary to ignore some of the molecules as they are chemically impossible 
                ignore+=1
                continue
        print(f"ignored: {ignore}")
        return np.stack(fingers)

    def data_tokens(self, max_len=None):
        tokens = []
        if self.qed == True:
            local = ["Surface_Area",
                     'ALERTS',
                     'ALOGP',
                     'AROM',
                     'HBA',
                     'HBD',
                     'MW',
                     'PSA',
                     'ROTB',
                     "moldb_smiles"]
            self.data.loc[:,local] = self.data.loc[:,local].astype(str)
            self.data['text'] = self.data[local].agg(' '.join, axis=1)
        elif self.qed == False:
            local = "moldb_smiles"
            self.data.loc[:,'text'] = self.data.loc[:,local]
        
        if max_len == None:
            self.max_len = self.data.text.str.len().max()+4
        else:
            self.max_len = max_len
        print(f"Length of longest string: {self.max_len}")
        for i, tok in enumerate(self.data.text):
            if len(tok) > self.max_len:
                print(len(tok))
            start = "%"
            stop = "|"
            pad = "_"
            if start in tok:
                print("unvaiable start token")
            if stop in tok:
                print("unvaiable stop token")
            if pad in tok:
                print("unvaiable pad token")
            tok = start+" "+tok+" "+stop
            tok = tok+pad*(self.max_len-len(tok)+1)
            token = [ord(char) for char in list(tok)] # ord turns single character strings into a ASCII unicode value as such it can be used as an automatic  
            ld = (float(self.data.loc[i,self.label]))
            ld = np.array(ld)
            token = np.array(token)
            tokens.append(np.append(ld,token))
        return np.stack(tokens)
    
    def properties(self, smile):
        mol = Chem.MolFromSmiles(str(smile))
        SA = MolSurf.LabuteASA(mol) # compute surface area
        props = list(QED.properties(mol)[:]) # compute QED 
        props.insert(0,SA)
        return props
    
    def tokenize(self,smile,max_len=None):
        ### Tokenize a single smile
        if max_len == None:
            max_len = self.max_len
        if self.qed != False:
            props = np.array(properties(smile))
            props = " ".join(props.astype(str))
            smile = props+smile
        start = "%"
        stop = "|"
        pad = "_"
        if start in smile:
            print("unvaiable start token")
        if stop in smile:
            print("unvaiable stop token")
        if pad in smile:
            print("unvaiable pad token")
        tok = start+" "+smile+" "+stop
        tok = tok+pad*(max_len-len(tok)+1)
        token = [ord(char) for char in list(tok)] 
        token = torch.unsqueeze(torch.tensor(token),0)
        return token
    
    def printfinger(self, smile, bits=128):
        ### Convert a single smile to a fingerprint
        finger = Chem.MolFromSmiles(smile)
        finger = np.asarray(Chem.GetMorganFingerprintAsBitVect(finger,2,bits))
        if self.qed==True:
            props = self.properties(smile)
            finger = np.append(finger,props)
        return finger
    
    def fingertokenize(self, smile, bits=128):
        ### Convert a singel smile to token and fingerprint for encoder-decoder transformer 
        finger = Chem.MolFromSmiles(smile)
        finger = np.asarray(Chem.GetMorganFingerprintAsBitVect(finger,2,bits))
        if self.qed==True:
            props = self.properties(smile)
            finger = np.append(finger,props)
        start = "%"
        stop = "|"
        pad = "_"
        if start in smile:
            print("unvaiable start token")
        if stop in smile:
            print("unvaiable stop token")
        if pad in smile:
            print("unvaiable pad token")
        tok = start+" "+smile+" "+stop
        tok = tok+pad*(max_len-len(tok)+1)
        token = [ord(char) for char in list(tok)] 
        token = torch.unsqueeze(torch.tensor(token),0)    
        fingertok = np.append(token,finger)
        return torch.tensor(fingertok)
    
    def make(self):
        # return the whole unbatched dataset
        return self.x, self.t
    
    def __getitem__(self, index):
        return self.x[index], self.t[index]
  
    def __len__(self):
        return len(self.data)
    
