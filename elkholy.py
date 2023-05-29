import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('8last_books.csv')







 # 1- DROP DUPLICATES
dataset.drop_duplicates(inplace=True)

# Calculate the total number of rows in the dataset
num_rows = len(dataset)

# Print the total number of rows
print("Total number of rows before delete of duplicates: ", num_rows)
 #--------------------------------------------------------------









# 2- CONVERT btshof ay sana belsaleb w btzawod 3aleha 1480
# apply the conversion to the desired column
dataset['original_publication_year'] = dataset['original_publication_year'].apply(lambda x: abs(x) + 1480 if x < 0 else x )





# check if there is a number bigger than 5 or less than 1 in the 'average_rating' column
if (dataset['average_rating'] > 5).any() or (dataset['average_rating'] < 1).any():
    print("There is a number bigger than 5 or less than 1 in the 'average_rating' column")
else:
    print("There is no number bigger than 5 or less than 1 in the 'average_rating' column")






# 3- NOT NULL
#btmla el missing values number  b elmean 
# Impute missing values in columns with integers

revenue = dataset['isbn']
revenue.head()
revenue_mean = revenue.mean()
revenue_mean
revenue.fillna(revenue_mean, inplace=True)
dataset.isnull().sum()
print(dataset.isnull().sum())







#btmla el missing values string b aktar klma mtkrraa
# Impute missing values in columns with strings
for col in dataset.select_dtypes(include='object' or 'str'):
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)
revenue = dataset['authors']
revenue.head()
revenue_mean = revenue.mode()
revenue_mean
revenue.fillna(revenue_mean, inplace=True)
dataset.isnull().sum()
print(dataset.isnull().sum())



#btmla el missing values string  b el values el adema fe title
dataset['original_title'] = dataset['original_title'].astype(str)
df_filtered = dataset.loc[dataset['original_title'].apply(type) == str]
print(dataset.dtypes)


# fill values in the 'column1' column that contain spaces with values from the 'column2' column
dataset.loc[dataset['original_title'].str.contains(' '), 'original_title'] = dataset['title']






# #--------------------------------------------------------------
 # drop rows missing data
null_rows = dataset.isnull().sum(axis=1).astype(bool).sum()

  # Print the number of null rows
print("Number of null rows before removing: ", null_rows)
dataset.dropna(inplace=True)





print(dataset.isnull().sum())



# #--------------------------------------------------------------

print(dataset.info())
#--------------------------------------------------------------
#Bysheel  el rmooz men el text column   -- 3- IN CONSISTENT FORMATS

#String
dataset['original_title'] = dataset['original_title'].str.replace('[^a-zA-Z0-9\s]', '')
dataset['title'] = dataset['title'].str.replace('[^a-zA-Z0-9\s]', '')
dataset['authors'] = dataset['authors'].str.replace('[^a-zA-Z0-9\s]', '')

#numbers
dataset['isbn'] = pd.to_numeric(dataset['isbn'], errors='coerce')
dataset['isbn13'] = pd.to_numeric(dataset['isbn13'], errors='coerce')
dataset['books_count'] = pd.to_numeric(dataset['books_count'], errors='coerce')
dataset['ratings_count'] = pd.to_numeric(dataset['ratings_count'], errors='coerce')
dataset['work_ratings_count'] = pd.to_numeric(dataset['work_ratings_count'], errors='coerce')
dataset['work_text_reviews_count'] = pd.to_numeric(dataset['work_text_reviews_count'], errors='coerce')
dataset['id'] = pd.to_numeric(dataset['id'], errors='coerce')
dataset['book_id'] = pd.to_numeric(dataset['book_id'], errors='coerce')
dataset['work_id'] = pd.to_numeric(dataset['work_id'], errors='coerce')


#--------------------------------------------------------------
# 4- Handle Caterogical Values




print(dataset.info())



dataset['isbn'] = pd.to_numeric(dataset['isbn'], errors='coerce')
dataset['isbn13'] = pd.to_numeric(dataset['isbn13'], errors='coerce')
dataset.dropna(inplace=True)
dataset['isbn'] = dataset['isbn'].astype(int)
print(dataset['isbn'].dtype)
print(dataset['isbn13'].dtype)

print(dataset.info)



# #--------------------------------------------------------------
# #--------------------------------------------------------------
# ##############################################################################################################################







# ### BADR QUESTION ############################################################################################################################
#  1- Which author has the most recent book in the dataset?


# sort the data by publication date
sorted_data = dataset.sort_values('original_publication_year', ascending=False)

# select the author of the most recent book
most_recent_author = sorted_data.iloc[0]['authors']

# print the result
print("The author of the most recent book is:", most_recent_author)


 

#2. What is the average number of pages for books in the dataset?
# Calculate the average number of pages
average_pages = dataset['books_count'].mean()

# Print the result
print("Average number of books:", average_pages)



## 3. When was the earliest book in the dataset published?
# sort the data by publication date
sorted_data = dataset.sort_values('original_publication_year')

# select the first row of the sorted data
earliest_book = sorted_data.iloc[0]

# extract the publication date of the earliest book
earliest_date = earliest_book['original_publication_year']

# print the result
print("The earliest book in the dataset was published in:", earliest_date)



##4. Which book in the dataset is the longest?
# the number of unique authors in the dataset
unique_authors = set(dataset['authors'])
num_authors = len(unique_authors)

print(f"There are {num_authors} unique authors in the dataset.")



##5. How many authors are represented in the dataset?
#  the book with the highest number of pages
longest_book = dataset.loc[dataset['books_count'].idxmax()]

print(f"The longest book is '{longest_book['title']}' with {longest_book['books_count']} pages.")
###############################################################################################################################





##############################################################################################################################
### MAHMOUD HOSSAM QUESTON ##########################################################################################################################

# (6-When was the latest book in the dataset published?) ?


# Calculate the latest publication year
latest_year = dataset['original_publication_year'].max()

# Print the latest publication year
print("The latest book in the dataset was published in:", latest_year)
# #----------------------------------------------------------------------------------------------------------------------------


# # 7. Which author has the most books in the dataset?


author_counts = dataset['authors'].value_counts()

# find the author with the most books
most_books_author = author_counts.index[0]

# print the author with the most books
print("The author with the most books in the dataset is:", most_books_author)


# #----------------------------------------------------------------------------------------------------------------------------

# 8. What is the total number of pages in the dataset?


# sum the number of pages for all books
total_pages = dataset['books_count'].sum()

# print the total number of pages
print("The total number of pages in the dataset is:", total_pages)

# #----------------------------------------------------------------------------------------------------------------------------


# 9. What is the median year of publication for books in the dataset?

# calculate the median publication year
median_year = dataset['original_publication_year'].median()

# print the median publication year
print("The median year of publication for books in the dataset is:", median_year)

# #----------------------------------------------------------------------------------------------------------------------------

# 10. Which author has the least number of books in the dataset?
# count the number of books by each author
author_counts = dataset['authors'].value_counts()

# find the author with the least number of books
least_books_author = author_counts.index[-1]

# print the author with the least number of books
print("The author with the least number of books in the dataset is:", least_books_author)






#----------------------------------------------------------------------------------------------------------------------------




### OMAR RAGAE QUESTION ############################################################################################################################

#--------- 11. How many books were published in the last decade (2010-2023) in the dataset?

# Load the dataset
books = pd.read_csv('books.csv')

# Convert the Publication Date column to a datetime object with only the year
books['original_publication_year'] = pd.to_datetime(books['original_publication_year'], format='%Y', errors='coerce')

# Define the start and end years of the decade
start_year = 2010
end_year = 2023

# Filter the books published within the decade
books_in_decade = books[(books['original_publication_year'].dt.year >= start_year) & (books['original_publication_year'].dt.year <= end_year)]

# Calculate the number of books published in the decade
num_books_in_decade = len(books_in_decade)

# Print the result
print(f"The number of books published between {start_year} and {end_year} is: {num_books_in_decade}")

#------------- 13. Which book in the dataset is the shortest?


# sort the DataFrame by the 'books_count' column in ascending order
dataset_sorted = dataset.sort_values(by='books_count', ascending=True)

# get the title of the shortest book from the 'original_title' column
shortest_book_title = dataset_sorted.iloc[0]['original_title']

# get the number of pages of the shortest book
shortest_book_pages = dataset_sorted.iloc[0]['books_count']

# print the title and number of pages of the shortest book
print("The shortest book is '{}' with {} pages.".format(shortest_book_title, shortest_book_pages))

#------ 14.How many pages do all the books by a specific author in the dataset have in total?

# specify the author to filter the DataFrame
author_name = 'John Green'

# filter the DataFrame to include only books by the specified author
author_books = dataset[dataset['authors'] == author_name]

# calculate the total number of pages for all books by the specified author
total_pages = author_books['books_count'].sum()

# print the total number of pages
print("The total number of pages for all books by {} is {}.".format(author_name, total_pages))

#--------------------- 15. What is the average publication year for books in the dataset?

# calculate the average publication year
avg_pub_year = dataset['original_publication_year'].mean()

# print the average publication year
print("The average publication year for books in the dataset is {:.0f}.".format(avg_pub_year))
###############################################################################################################################



#### SAMIR QUESTION ###########################################################################################################################
# # What is the average number ofbooks per author in the dataset ? 



# # Group the dataset by the "authors" column and count the number of books for each author
book_counts = dataset.groupby('authors').size()

# Calculate the average number of books per author
avg_books_per_author = book_counts.mean()

# Print the average number of books per author
print("The average number of books per author in the dataset is: ", round(avg_books_per_author))



# How many books in the dataset were published before a certain year (e.g. before 2000) ? 


# Set the year
year = 2000

# Filter the dataset to include only the books published before the year
filtered_dataset = dataset.loc[dataset['original_publication_year'] < year]

# Count the number of rows in the filtered dataset
num_books_before_year = filtered_dataset.shape[0]

# Print the number of books in the dataset published before the year
print("The number of books in the dataset published before ", year, " is: ", num_books_before_year)



# # Which author has the highest number of books published in a single year in the dataset?

# Group the dataset by the "authors" and "publication_year" columns and count the number of books for each author-year combination
book_counts = dataset.groupby(['authors', 'original_publication_year']).size()

# Find the maximum count
max_count = book_counts.max()

# Filter the dataframe to include only the rows with the maximum count
max_count_df = book_counts[book_counts == max_count]

# Print the author and year for the rows with the maximum count
for index, value in max_count_df.items():
    author, year = index
    print("The author with the highest number of books published in a single year is ", author, " in the year ", year, " with ", max_count, " books.")




# # What is the total number of pages for books published by a specific author in the dataset?


# Load the dataset
books = pd.read_csv('books.csv')



# 1. Filter the dataset to include only the books published by the specific author
author = "John Green"
author_books = books[books['authors'] == author]

# 2. Calculate the total number of pages for all the books published by the author
total_pages = author_books['books_count'].sum()

# Print the total number of pages for books published by John Green
print(f"The total number of pages for books published by {author}: {total_pages}")

# What is the most common year of publication for books in the dataset?

# Count the frequency of each unique value in the publication_year column
year_counts = books['original_publication_year'].value_counts()

# Find the most common year of publication
most_common_year = year_counts.idxmax()

# Print the most common year of publication
print(f"The most common year of publication for books in the dataset is: {most_common_year}")




# # Which author has the lowest average number of pages per book in the dataset?

# # Calculate the average number of pages per book for each author
avg_pages_per_book = dataset.groupby('authors')['books_count'].mean()

# Sort the series in ascending order
avg_pages_per_book = avg_pages_per_book.sort_values()

# Print the author with the lowest average number of pages per book
print("Author with the lowest average number of pages per book:", avg_pages_per_book.index[0])



###############################################################################################################################
## ELKHOLY QUESTIONS #############################################################################################################################
#(//26-How many books in the dataset have more than a certain number of books count (e.g. more than 500 books count) )?
# Load the dataset
books = pd.read_csv('books.csv')

# Count the number of books with more than 500 books count
num_books = len(books[books['books_count'] > 500])

print("The number of books in the dataset with more than 500 books_count is:", num_books)



###############################################################################################################################
# //(27-Which book in the dataset has the most recent publication date?) ?
# Load the dataset
books = pd.read_csv('books.csv')

# Convert the Publication Date column to a datetime object
books['original_publication_year'] = pd.to_datetime(books['original_publication_year'], errors='coerce')

# Find the book with the most recent publication date
most_recent_book = books.loc[books['original_publication_year'].idxmax(), 'title']

print("The book with the most recent publication date is:", most_recent_book)



###############################################################################################################################
# (//28-What is the average number of years between publication dates for books by a specific author in the dataset?)?

# Load the dataset
books = pd.read_csv('books.csv')

# Convert the Publication Date column to a datetime object with only the year
books['original_publication_year'] = pd.to_datetime(books['original_publication_year'], format='%Y', errors='coerce')

# Choose a specific author
author = 'John Green'

# Filter the dataset to include only books by the chosen author
author_books = books.loc[books['authors'] == author]

# Calculate the average number of years between publication dates
if len(author_books) >= 2:
    avg_years = (author_books['original_publication_year'].max().year - author_books['original_publication_year'].min().year) / (len(author_books) - 1)
    print(f"The average number of years between publication dates for books by {author} is: {avg_years:.2f}")



###############################################################################################################################
# //(29-Which author in the dataset has the most diverse range of publication dates?)?

# Load the dataset
books = pd.read_csv('books.csv')

# Convert the Publication Date column to a datetime object
books['original_publication_year'] = pd.to_datetime(books['original_publication_year'], errors='coerce')

# Group the dataset by author
grouped = books.groupby('authors')

# Calculate the range of publication dates for each author
date_range = grouped['original_publication_year'].agg(lambda x: (x.max() - x.min()).days)

# Find the author with the most diverse range of publication dates
most_diverse_author = date_range.idxmax()

print("The author with the most diverse range of publication dates is:", most_diverse_author)



###############################################################################################################################
## //(30. What is the total number of pages for books published in a specific year in the dataset?)?

# Load the dataset
books = pd.read_csv('books.csv')

# Choose a specific year
year = 2012

# Filter the dataset to include only books published in the chosen year
year_books = books.loc[books['original_publication_year'] == year]

# Calculate the total number of pages for books published in the chosen year
total_pages = year_books['books_count'].sum()

print(f"The total number of books count for books published in {year} is: {total_pages}")

##############################################################################################################################


##############################################################################################################################



# the outlier---------
# calculate the interquartile range (IQR) of the average rating column
Q1 = dataset['average_rating'].quantile(0.25)
Q3 = dataset['average_rating'].quantile(0.75)
IQR = Q3 - Q1

# identify any values that are outside the range of 1.5 times the IQR below Q1 or above Q3
outliers = dataset[(dataset['average_rating'] < Q1 - 1.5 * IQR) | (dataset['average_rating'] > Q3 + 1.5 * IQR)]

# print the outliers
print("The outliers of the average rating column are:")
print(outliers)

dataset.to_csv('66last_books.csv', index=False)







# code bymas7 column mo3ayn isbn13
# Drop the specific column using the .drop() method
df = dataset.drop('isbn13', axis=1)

# Display the updated dataframe
print(df)
































