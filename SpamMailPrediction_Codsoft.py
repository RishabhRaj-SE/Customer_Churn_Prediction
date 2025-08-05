{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN3oL3QSSHQFiOZRpccU6SC",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/RishabhRaj-SE/Customer_Churn_Prediction/blob/main/SpamMailPrediction_Codsoft.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "YF2ALpqnUb4_"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "raw_mail_data=pd.read_csv('/content/spam.csv')"
      ],
      "metadata": {
        "id": "9A2Dh3CsWEE8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "raw_mail_data.head()\n"
      ],
      "metadata": {
        "collapsed": true,
        "id": "HenW-TqtWbVH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "raw_mail_data.shape"
      ],
      "metadata": {
        "id": "Z8hVGxuLWlRW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "raw_mail_data.value_counts().unique()"
      ],
      "metadata": {
        "id": "D_3fUxtRWrUI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(raw_mail_data.columns)"
      ],
      "metadata": {
        "id": "Jet57gSOW0ei"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')"
      ],
      "metadata": {
        "id": "2Oun0WS1XAZ0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mail_data.head()"
      ],
      "metadata": {
        "id": "84xW1igNXYnx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# label spam mail as 0;  ham mail as 1;\n",
        "\n",
        "mail_data.loc[mail_data['v1'] == 'spam', 'v1',] = 0\n",
        "mail_data.loc[mail_data['v1'] == 'ham', 'v1',] = 1"
      ],
      "metadata": {
        "id": "WTqNBQsQXcJE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X=mail_data['v2']\n",
        "Y=mail_data['v1']"
      ],
      "metadata": {
        "id": "QsDSOK4IYNx3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(X)\n",
        "print(Y)"
      ],
      "metadata": {
        "id": "ZkwRym09YSHA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)"
      ],
      "metadata": {
        "id": "na5nYAWiYbeJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(X.shape)\n",
        "print(X_train.shape)\n",
        "print(X_test.shape)"
      ],
      "metadata": {
        "id": "u_-aHYy9YjJE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# transform the text data to feature vectors that can be used as input to the Logistic regression\n",
        "\n",
        "feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)\n",
        "\n",
        "X_train_features = feature_extraction.fit_transform(X_train)\n",
        "X_test_features = feature_extraction.transform(X_test)\n",
        "\n",
        "# convert Y_train and Y_test values as integers\n",
        "\n",
        "Y_train = Y_train.astype('int')\n",
        "Y_test = Y_test.astype('int')"
      ],
      "metadata": {
        "id": "pPun7KtFaZEZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = LogisticRegression()"
      ],
      "metadata": {
        "id": "6zK4ZAqva66f"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit(X_train_features, Y_train)"
      ],
      "metadata": {
        "id": "F0IIuFIoa8Rp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# prediction on training data\n",
        "\n",
        "prediction_on_training_data = model.predict(X_train_features)\n",
        "accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)"
      ],
      "metadata": {
        "id": "2DpgieQHbZ7Y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print('Accuracy on training data : ', accuracy_on_training_data)"
      ],
      "metadata": {
        "id": "DiRd_XBebh-4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "prediction_on_test_data = model.predict(X_test_features)\n",
        "accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)"
      ],
      "metadata": {
        "id": "6JMKw_qebm3f"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print('Accuracy on test data : ', accuracy_on_test_data)"
      ],
      "metadata": {
        "id": "wfLU83Mfbs4B"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "input_mail = [\"Free,Free,Free.Book free tickets of WorldCup Match for free\"]\n",
        "\n",
        "# convert text to feature vectors\n",
        "input_data_features = feature_extraction.transform(input_mail)\n",
        "\n",
        "# making prediction\n",
        "\n",
        "prediction = model.predict(input_data_features)\n",
        "print(prediction)\n",
        "\n",
        "\n",
        "if (prediction[0]==1):\n",
        "  print('Ham mail')\n",
        "\n",
        "else:\n",
        "  print('Spam mail')"
      ],
      "metadata": {
        "id": "1zDYneyTb7J9"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}