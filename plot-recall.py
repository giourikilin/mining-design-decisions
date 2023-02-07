import matplotlib.pyplot as plt

# Define the number of items in the search and the retrieved items
retrieved_items = [20, 40, 60, 80, 100]

# Calculate the recall for each number of retrieved items
# recall = [i / number_of_items for i in retrieved_items]

#recall1 = [0.7, 0.725, 0.8, 0.6875, 0.69]

recall2 = [0.55, 0.58, 0.66, 0.63, 0.56]

# Plot the recall
plt.plot(retrieved_items, recall2)
plt.xlabel('Number of Retrieved Items')
plt.ylabel('Recall')
plt.title('Recall of Search engine (20 results per page)')
plt.show()