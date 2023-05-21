#!/usr/bin/env python
# coding: utf-8

# In[6]:


import random
import pandas as pd

UserID = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

print(UserID)
print(len(UserID))


# In[7]:


import random
import pandas as pd

Email = ["AA@gmail.com", "AB@gmail.com", "AC@gmail.com", "AD@gmail.com", "AE@gmail.com", "AF@gmail.com", "AG@gmail.com", "AH@gmail.com", "AI@gmail.com", "AJ@gmail.com", "AK@gmail.com", "AL@gmail.com", "AM@gmail.com", "AN@gmail.com", "AO@gmail.com", "AP@gmail.com", "AQ@gmail.com", "AR@gmail.com", "AS@gmail.com", "AT@gmail.com"]

print(Email)
print(len(Email))


# In[8]:


import random
import pandas as pd

firstName = ["AAF", "ABF", "ACF", "ADF", "AEF", "AFF", "AGF", "AHF", "AIF", "AJF", "AKF", "ALF", "AMF", "ANF", "AOF", "APF", "AQF", "ARF", "ASF", "ATF"]
lastName = ["AAL", "ABL", "ACL", "ADL", "AEL", "AFL", "AGL", "AHL", "AIL", "AJL", "AKL", "ALL", "AML", "ANL", "AOL", "APL", "AQF", "ARF", "ASF", "ATF"]

shuffled_firstName = firstName.copy()
shuffled_lastName = lastName.copy()

random.shuffle(shuffled_firstName)
random.shuffle(shuffled_lastName)

print(shuffled_firstName)
print(shuffled_lastName)
print(len(shuffled_firstName))
print(len(shuffled_lastName))


# In[9]:


name = list(zip(shuffled_firstName, shuffled_lastName))

print(name)
print(len(name))


# In[10]:


import random

shuffled_firstNames = [name[0] for name in name]
shuffled_lastNames = [name[1] for name in name]

# Define function to mask the elements of a list
def mask_list_elements(input_list, mask_char):
    """
    This function masks the elements of a list with a given character.
    """
    count = len(input_list)
    cnt = 0
    output_list = []
    for item in input_list:
        if(cnt < count) :
            output_list.append(mask_char * (len(item)-1) + item[-1])
            #output_list.append(item[-1])
            cnt = cnt + 1
    return output_list

# Example usage

masked_shuffled_firstNames = mask_list_elements(shuffled_firstNames, '*')
masked_shuffled_lastNames = mask_list_elements(shuffled_lastNames, '*')
print(masked_shuffled_firstNames)
print(masked_shuffled_lastNames)
print(len(masked_shuffled_firstNames))
print(len(masked_shuffled_lastNames))

maskedName = list(zip(masked_shuffled_firstNames, masked_shuffled_lastNames))

print(maskedName)
print(len(maskedName))


# In[11]:


import random
import pandas as pd

Gender = ['M','M','M','M','M','F','F','F','F','F','M','M','M','M','M','F','F','F','F','F']

print(Gender)
print(len(Gender))


# In[12]:


import random
import pandas as pd

Age = [22,26,28,25,20,46,45,47,41,48,23,24,27,29,21,44,43,49,40,42]

print(Age)
print(len(Age))


# In[13]:


import random
import pandas as pd

Country = ['Argentina','Bolivia','Brazil','Chile','Colombia','Ecuador','Guyana','Paraguay','Peru','Uruguay','China','India','Indonesia','Pakistan','Bangladesh','Japan','Philippines','Vietnam','Turkey','Iran']

print(Country)
print(len(Country))


# In[14]:


import random
import pandas as pd

Relationship_Status = ['Single','Single','Single','Single','Single','Married','Married','Married','Married','Married','Divorced','Divorced','Divorced','Divorced','Divorced','Single','Single','Single','Single','Single']

print(Relationship_Status)
print(len(Relationship_Status))


# In[15]:


import random
import pandas as pd

Chat_Sharing_Preferences = ["Don't Share", "Share with therapist only", "Use for targeted advertisment", "Use for research purposes", "Share with third-parties"]

Final_Chat_Sharing_Preferences = []

for i in range(1, 21) :
    CSP = random.choice(Chat_Sharing_Preferences)
    Final_Chat_Sharing_Preferences.append(CSP)
    
print(Final_Chat_Sharing_Preferences)
print(len(Final_Chat_Sharing_Preferences))


# In[16]:


import random
import pandas as pd

DocumentsSharedWithTherapist = ["Medical history", "Personal journal or diary", "Treatment plan or goals", "Worksheets or exercises"] 

Final_DocumentsSharedWithTherapist = []

listFDSWT = [1,2,3,4,5,6,7,8,9,10] 

for i in range(1, 21) :
    FS = random.choice(DocumentsSharedWithTherapist)
    Final_DocumentsSharedWithTherapist.append(FS)
    
print(Final_DocumentsSharedWithTherapist)
print(len(Final_DocumentsSharedWithTherapist))


# In[17]:


import random
import pandas as pd

Symptoms = ["Changes in mood", "Changes in behavior", "Cognitive changes", "Physical symptoms", "Thoughts of suicide or self-harm"]

Final_Symptoms = []

listFS = [1,2,3,4,5,6,7,8,9,10] 

for i in range(1, 21) :
    FS = random.choice(Symptoms)
    Final_Symptoms.append(FS)    
    
print(Final_Symptoms)
print(len(Final_Symptoms))


# In[22]:


columns = ["UserID", "Email", "Name", "Country", "Age", "Gender", "RelationshipStatus", "Chat Sharing Preferences", "Documents Shared With Therapist", "Symptoms"]
data = []

AgeRange = []
for i in range(20) :
    if Age[i] >= 20 and Age[i] <= 29:
        AgeRange.append('2*')
    elif Age[i] >= 40 and Age[i] <= 49:
        AgeRange.append('4*')

print(AgeRange)
print(len(AgeRange))

GenderRange = []
for i in range(20) :
    GenderRange.append('*')

print(GenderRange)
print(len(GenderRange))

Relationship_StatusRange = []

for i in range(20) :
    if Relationship_Status[i] == 'Single' :
        Relationship_StatusRange.append('S*')
    if Relationship_Status[i] == 'Married' :
        Relationship_StatusRange.append('M*')        
    if Relationship_Status[i] == 'Divorced' :
        Relationship_StatusRange.append('D*')        

print(Relationship_StatusRange)
print(len(Relationship_StatusRange))

#Country = ['China','India','Pakistan','Sri Lanka','Italy','Spain','France','United Kingdom','Egypt','South Africa','Nigeria','Sudan','Chile','Argentina','Brazil','Ecuador']
CountryRange = []

for i in range(20) :
    if Country[i] == 'China' :
        CountryRange.append('Asia')
    elif Country[i] == 'India' :
        CountryRange.append('Asia')        
    elif Country[i] == 'Pakistan' :
        CountryRange.append('Asia')
    elif Country[i] == 'Indonesia' :
        CountryRange.append('Asia')
    elif Country[i] == 'Bangladesh' :
        CountryRange.append('Asia')
    elif Country[i] == 'Japan' :
        CountryRange.append('Asia')        
    elif Country[i] == 'Philippines' :
        CountryRange.append('Asia')
    elif Country[i] == 'Vietnam' :
        CountryRange.append('Asia')
    elif Country[i] == 'Turkey' :
        CountryRange.append('Asia')
    elif Country[i] == 'Iran' :
        CountryRange.append('Asia')           
        
    elif Country[i] == 'Argentina' :
        CountryRange.append('South America')
    elif Country[i] == 'Bolivia' : 
        CountryRange.append('South America')
    elif Country[i] == 'Brazil' : 
        CountryRange.append('South America')
    elif Country[i] == 'Chile' :
        CountryRange.append('South America')
    elif Country[i] == 'Colombia' :
        CountryRange.append('South America')
    elif Country[i] == 'Ecuador' : 
        CountryRange.append('South America')
    elif Country[i] == 'Guyana' : 
        CountryRange.append('South America')
    elif Country[i] == 'Paraguay' :
        CountryRange.append('South America')
    elif Country[i] == 'Peru' :
        CountryRange.append('South America')
    elif Country[i] == 'Uruguay' : 
        CountryRange.append('South America')        
             
print(CountryRange)
print(len(CountryRange))

for i in range(20):
    row = [UserID[i], Email[i], maskedName[i], CountryRange[i], AgeRange[i], GenderRange[i], Relationship_StatusRange[i], Final_Chat_Sharing_Preferences[i], Final_DocumentsSharedWithTherapist[i], Final_Symptoms[i]]
    data.append(row)
    
print(data)
print(len(data))


# In[23]:


for row in data:
    for i in range(len(columns)):
        print(columns[i] + ": " + str(row[i]))
    print(" ")


# In[24]:


import random

random.shuffle(data)

print(data)


# In[25]:


for row in data:
    for i in range(len(columns)):
        print(columns[i] + ": " + str(row[i]))
    print(" ")


# In[26]:


df_Table = pd.DataFrame(data, columns= ["UserID", "Email", "Name", "Country", "Age", "Gender", "RelationshipStatus", "Chat Sharing Preferences", "Documents Shared With Therapist", "Symptoms"])

print(df_Table)

df_Table['Name'] = maskedName


# In[27]:


import pandas as pd

df_sorted = df_Table.sort_values('Country', ascending=True)

print(df_sorted)


# In[28]:


import pandas as pd

df_sorted2 = df_Table.sort_values(by=['Country', 'Age', 'Gender', 'RelationshipStatus'], ascending=True)

print(df_sorted2)


# In[31]:


import pandas as pd

def k_anonymity(df_Table, columns, k):
    """
    Apply k-anonymity to a Pandas DataFrame.

    Parameters:
    - df: the input DataFrame.
    - columns: a list of column names to apply k-anonymity to.
    - k: the desired level of k-anonymity.

    Returns:
    - The anonymized DataFrame.
    """
    df_anon = df_Table.copy()

    # Group by the specified columns and count the number of records in each group.
    group_counts = df_anon.groupby(columns).size().reset_index(name='count')

    # Filter out groups with less than k records.
    group_counts = group_counts[group_counts['count'] >= k]

    # Merge the filtered group counts back into the original DataFrame.
    df_anon = pd.merge(df_anon, group_counts, on=columns, how='inner')

    # Drop the count column.
    df_anon.drop(columns=['count'], inplace=True)

    return df_anon

anon_df = k_anonymity(df_Table, ['Country', 'Age', 'Gender', 'RelationshipStatus'], 5)

print(anon_df)
print(len(anon_df))


# In[ ]:




