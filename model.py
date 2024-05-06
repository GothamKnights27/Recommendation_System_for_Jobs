import pandas as pd
import re
import nltk
import os
nltk.download('stopwords')
from nltk.corpus import stopwords
stopw  = set(stopwords.words('english'))

def predict(resume):

    unclean_df = pd.read_csv('DataAnalyst.csv')

    # Data Cleaning
    unclean_df.dropna(inplace = True)

    def convert_revenue(value):
        if 'Unknown' in value:
            return None
        elif ' to ' in value:
            values = re.findall(r'\d+\.?\d*', value)
            min_revenue = float(values[0])
            max_revenue = float(values[1])
            unit = value.split()[-2]
            if unit == 'billion':
                min_revenue *= 1000
                max_revenue *= 1000
            return (min_revenue + max_revenue) / 2
        else:
            numerical_values = re.findall(r'\d+\.?\d*', value)
            if numerical_values:
                return float(numerical_values[0])
            else:
                return None

    # Apply the conversion function to the "Revenue" column
    unclean_df['Average Revenue'] = unclean_df['Revenue'].apply(convert_revenue)

    unclean_df.dropna(inplace = True)

    unclean_df['Easy Apply'] = unclean_df['Easy Apply'].replace('-1', 'False')

    # Convert the "Size" column to string type
    unclean_df['Size'] = unclean_df['Size'].astype(str)

    # Define a function to convert the size value
    def convert_size(value):
        if 'Unknown' in value:
            return None
        elif ' to ' in value:
            sizes = value.split(' to ')
            min_size = int(sizes[0].replace('+', '').replace(',', '').split()[0])
            max_size = int(sizes[1].replace('+', '').replace(',', '').split()[0])
            return (min_size + max_size) / 2
        else:
            return int(value.replace('+', '').replace(',', '').split()[0])

    # Apply the conversion function to the "Size" column
    unclean_df['Size'] = unclean_df['Size'].apply(convert_size)

    unclean_df.dropna(inplace = True)

    unclean_df['Processed_JD']=unclean_df['Job Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word)>2 and word not in (stopw)]))

    def convert_salary(value):
        if 'Unknown' in value:
            return None
        elif '-' in value:
            values = re.findall(r'\$\d+K', value)
            min_value = int(values[0].replace('$', '').replace('K', '')) if values else None
            max_value = int(values[1].replace('$', '').replace('K', '')) if len(values) > 1 else None
            if min_value and max_value:
                return (min_value + max_value) / 2
            elif min_value:
                return min_value
            elif max_value:
                return max_value
            else:
                return None
        else:
            return int(re.findall(r'\$\d+K', value)[0].replace('$', '').replace('K', ''))
        
    unclean_df['Average Salary'] = unclean_df['Salary Estimate'].apply(convert_salary)

    unclean_df.dropna(inplace = True)

    unclean_df.reset_index(drop = True, inplace = True)

    unclean_df['Company Name'] = unclean_df['Company Name'].apply(lambda x: x.split('\\')[0] if '\\' in x else x)

    clean_df = unclean_df.drop(['Unnamed: 0', 'Salary Estimate', 'Job Description', 'Headquarters', 'Revenue', 'Competitors'], axis = 1)


    # Feature Engineering
    from ftfy import fix_text
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from docx import Document
    from skills_extraction import skills_extractor

    # Now proceed with your skills extraction
    skills = []
    try:
        skills.append(' '.join(word for word in skills_extractor(resume)))
    except Exception as e:
        print("Error:", e)

    def ngrams(string, n=3):
        string = fix_text(string) # fix text
        string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
        string = string.lower()
        chars_to_remove = [")","(",".","|","[","]","{","}","'"]
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        string = re.sub(rx, '', string)
        string = string.replace('&', 'and')
        string = string.replace(',', ' ')
        string = string.replace('-', ' ')
        string = string.title() # normalise case - capital at start of each word
        string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
        string = ' '+ string +' ' # pad names for ngrams...
        string = re.sub(r'[,-./]|\sBD',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform(skills)

    def getNearestN(query):
        queryTFIDF_ = vectorizer.transform(query)
        distances, indices = nbrs.kneighbors(queryTFIDF_)
        return distances, indices

    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    jd_test = (clean_df['Processed_JD'].values.astype('U'))

    distances, indices = getNearestN(jd_test)
    test = list(jd_test)
    matches = []

    for i,j in enumerate(indices):
        dist=round(distances[i][0],2)

        temp = [dist]
        matches.append(temp)

    matches = pd.DataFrame(matches, columns=['Match Confidence'])

    clean_df['Match Confidence']=matches['Match Confidence']

    recommendation_df = clean_df.sort_values(by = 'Match Confidence', ascending = False)

    recommendation_df_final = recommendation_df.head()

    return recommendation_df_final