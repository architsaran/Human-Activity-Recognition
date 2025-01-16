#!/usr/bin/env python
# coding: utf-8

# # Tutorial to test different filtering techniques and ML models

# ### Importing The Necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats as stats
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
#get_ipython().run_line_magic('matplotlib', 'inline')


# ### Rename the CSV file as per activity and load into the respective dataframe

# In[2]:


file_path_sit = 'Sit_Data.csv'
file_path_walk = 'Walk_Data.csv'
file_path_run = 'Run_Data.csv'
data_sit = pd.read_csv(file_path_sit)
data_walk = pd.read_csv(file_path_walk)
data_run = pd.read_csv(file_path_run)


# ### Display the first few rows of the data

# In[3]:


data_sit.insert(0,"Activity","Sitting")
data_walk.insert(0,"Activity","Walking")
data_run.insert(0,"Activity","Running")


# ### Checking if data can be extracted activity wise

# In[4]:


raw_data = pd.concat([data_sit, data_walk, data_run], ignore_index=True)
raw_data[raw_data['Activity']=='Sitting']


# ### Checking if the recorded is balanced

# In[5]:


raw_data['Activity'].value_counts()


# In[6]:


Activities = raw_data['Activity'].value_counts().index
Fs=50
Activities


# ### Plotting and visualising the accelerometer data

# In[7]:


#x=raw_data["Linear Acceleration x (m/s^2)"]
#y=raw_data["Linear Acceleration y (m/s^2)"]
#z=raw_data["Linear Acceleration z (m/s^2)"]
#timestamp=raw_data["Time (s)"]
def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 7), sharex=True)
    plot_axis(ax0, data["Time (s)"], data["Linear Acceleration x (m/s^2)"], 'X-Axis')
    plot_axis(ax1, data["Time (s)"], data["Linear Acceleration y (m/s^2)"], 'Y-Axis')
    plot_axis(ax2, data["Time (s)"], data["Linear Acceleration y (m/s^2)"], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

for activity in Activities:
    data_for_plot = raw_data[(raw_data['Activity'] == activity)][:Fs*100]
    plot_activity(activity, data_for_plot)


# ### Checking if the hardware is recording data properly.

# In[8]:


tsdiff= data_run['Time (s)'].diff()
tsvar=tsdiff.var()
print(tsdiff)
print(tsvar)
print(1/np.mean(tsdiff))


# The variance in timestep is negligible and the hardware performs up to expectations

# ### Displaying the frequency content from an activity

# In[9]:


sampling_rate = 50
sig=raw_data[raw_data["Activity"]=="Sitting"]['Linear Acceleration x (m/s^2)'].fillna(0)
#sig2=filtered_data[filtered_data["Activity"]=="Running"]['x_sma'].fillna(0)
def plot_frequency_spectrum(column, fs, step=5):
    
    # Load the data
    data = column

    # Extract the signal data from the specified column
    signal = data.values
    
    sampling_rate = fs

    # Apply FFT
    n = len(signal)
    frequency = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_values = np.fft.fft(signal)

    # Only keep the positive frequencies (real signal is symmetric)
    frequency = frequency[:n//2]
    fft_values = np.abs(fft_values[:n//2])

    # Convert FFT amplitude to decibels (dB)
    fft_values_db = 20 * np.log10(fft_values)

    # Plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(frequency, fft_values_db)
    plt.title(f"Frequency Spectrum of axis Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(True)

    # Add finer x-axis ticks
    plt.xticks(np.arange(0, np.max(frequency), step=step))  # Adjust the step size as needed

    # Show the plot
    plt.show()

def plot_spectrogram(column, fs, nperseg=256, noverlap=128):

    # Load the data
    data = column

    # Extract the signal data from the specified column
    signal = data.values
    
    sampling_rate = fs

    # Generate the spectrogram
    frequencies, times, Sxx = scipy.signal.spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.title(f"Spectrogram of axis Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power/Frequency (dB/Hz)")
    plt.ylim(0, np.max(frequencies))
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_periodogram(column, fs):
    

    # Load the data
    data = column

    # Extract the signal data from the specified column
    signal = data.values

   
    sampling_rate = fs

    # Generate the periodogram
    frequencies, psd = scipy.signal.periodogram(signal,sampling_rate)

    # Convert PSD to decibels
    psd_db = 10 * np.log10(psd)

    # Plot the periodogram
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, psd_db)
    plt.title(f"Periodogram (PSD) of axis Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.grid(True)

    # Add finer x-axis ticks
    plt.xticks(np.arange(1, np.max(frequencies), step=1))  # Adjust the step size as needed

    # Show the plot
    plt.show()
    
def plot_welch(column, fs, beta=14.0, nperseg=256):
    
    # Load the data
    data = column

    # Extract the signal data from the specified column
    signal = data.values

    
    sampling_rate = fs

    # Create the Kaiser window
    window = scipy.signal.kaiser(nperseg, beta=beta)

    # Apply Welch's method with 50% overlap and Kaiser window
    frequencies, psd = scipy.signal.welch(signal, fs=sampling_rate, window=window, nperseg=nperseg, noverlap=nperseg // 2)

    # Convert PSD to decibels
    psd_db = 10 * np.log10(psd)

    # Plot the PSD
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, psd_db)
    plt.title(f"Welch PSD of axis Data with Kaiser Window and 50% Overlap")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.grid(True)

    # Add finer x-axis ticks
    plt.xticks(np.arange(0, np.max(frequencies), step=1))  # Adjust the step size as needed

    # Show the plot
    plt.show()

plot_frequency_spectrum(sig, sampling_rate, step=1)


# In[10]:


plot_periodogram(sig, sampling_rate)


# In[11]:


plot_spectrogram(sig, sampling_rate, nperseg=256, noverlap=128)


# In[12]:


plot_welch(sig, sampling_rate, beta=14.0, nperseg=256)


# #### Defining a Simple Moving Average Filter

# In[13]:


def moving_average_filter(data, window_size=5):
    return data.rolling(window=window_size).mean()


# #### Defining a Butterworth Filter

# In[14]:


def butterworth_filter(data, cutoff=10, fs=50.0, order=30):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


# ### Applying the filters

# In[15]:


filtered_data=raw_data
filtered_data['x_sma'] = moving_average_filter(raw_data["Linear Acceleration x (m/s^2)"])
filtered_data['y_sma'] = moving_average_filter(raw_data["Linear Acceleration y (m/s^2)"])
filtered_data['z_sma'] = moving_average_filter(raw_data["Linear Acceleration z (m/s^2)"])


filtered_data['x_butter'] = butterworth_filter(raw_data["Linear Acceleration x (m/s^2)"])
filtered_data['y_butter'] = butterworth_filter(raw_data["Linear Acceleration y (m/s^2)"])
filtered_data['z_butter'] = butterworth_filter(raw_data["Linear Acceleration z (m/s^2)"])


# ### Plot the filtered data

# In[16]:


avsig=filtered_data[filtered_data["Activity"]=="Sitting"]['x_sma'].fillna(0)
plot_welch(avsig, sampling_rate, beta=14.0, nperseg=256)


# In[17]:


btsig=filtered_data[filtered_data["Activity"]=="Sitting"]['x_butter'].fillna(0)
plot_welch(avsig, sampling_rate, beta=14.0, nperseg=256)


# In[19]:


def plot_3d_scatter(table, x_col, y_col, z_col, label_col):
    
    # Load the data
    data = table

    # Extract the columns for X, Y, Z, and activity label
    x_data = data[x_col]
    y_data = data[y_col]
    z_data = data[z_col]
    labels = data[label_col]

    # Get the unique activity labels
    unique_labels = labels.unique()

    # Set up the 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each activity group with a different color
    for label in unique_labels:
        # Extract the data corresponding to this activity label
        idx = labels == label
        ax.scatter(x_data[idx], y_data[idx], z_data[idx], label=label)

    # Set plot labels and title
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    plt.title('3D Scatter Plot Grouped by Activity Label')
    ax.view_init(elev=90, azim=90)

    # Add a legend
    plt.legend(title=label_col)

    # Show the plot
    plt.show()

plot_3d_scatter(raw_data, "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)", "Activity")


# In[20]:


plot_3d_scatter(filtered_data, "x_sma", "y_sma", "z_sma", "Activity")


# In[21]:


plot_3d_scatter(filtered_data, "x_butter", "y_butter", "z_butter", "Activity")


# In[22]:


filtered_data.head(-5)


# ### Defining the Dependent and Independent variables

# In[23]:


X = raw_data[['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)']]
y = raw_data['Activity']


# ### Standardizing the data

# In[24]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


X


# ### Splitting the data into Training data and Testing data

# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Verifying the data is as close to equally split as possible

# In[26]:


y_train.value_counts()


# In[27]:


X.shape, y.shape


# In[28]:


X_train.shape, X_test.shape


# ### Fitting the model and evaluating the performance

# In[29]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))


# In[30]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation and get accuracy scores for each fold
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Print the accuracy scores for each fold
print(f"Cross-Validation Scores: {scores}")
    
# Print the mean and standard deviation of the accuracy scores
print(f"Mean Accuracy: {scores.mean()}")
print(f"Standard Deviation: {scores.std()}")


# In[31]:


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=y.value_counts().index, show_normed=True, figsize=(7,7))


# In[32]:


from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

def train_and_evaluate_model(X, y, model):
    #Standardize the values of x
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Print classification report and accuracy score
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred))
    
    mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat=mat, class_names=y.value_counts().index, show_normed=True, figsize=(7,7))
    
    # Perform cross-validation and get accuracy scores for each fold
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    # Print the accuracy scores for each fold
    print(f"Cross-Validation Scores: {scores}")
    
    # Print the mean and standard deviation of the accuracy scores
    print(f"Mean Accuracy: {scores.mean()}")
    print(f"Standard Deviation: {scores.std()}")


# In[33]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
X = filtered_data[['x_sma', 'y_sma', 'z_sma']].fillna(0)
y = filtered_data['Activity']

train_and_evaluate_model(X, y, model)


# In[34]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
X = filtered_data[['x_sma', 'y_sma', 'z_sma']].fillna(0)
y = filtered_data['Activity']

train_and_evaluate_model(X, y, model)


# In[35]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
X = filtered_data[['x_sma', 'y_sma', 'z_sma']].fillna(0)
y = filtered_data['Activity']

train_and_evaluate_model(X, y, model)


# In[64]:


X = filtered_data[['x_sma', 'y_sma', 'z_sma']].fillna(0)
y = filtered_data['Activity']
model = RandomForestClassifier(n_estimators=100, random_state=42)

train_and_evaluate_model(X, y, model)


# In[36]:


X = filtered_data[['x_butter', 'y_butter', 'z_butter']]
y = filtered_data['Activity']
model = RandomForestClassifier(n_estimators=100, random_state=42)

train_and_evaluate_model(X, y, model)

