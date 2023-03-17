from flask import Flask,request,app,jsonify,url_for,render_template
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    form_data = request.form.to_dict()
    CustomerID = form_data.get('CustomerID')
    Name = form_data.get('Name')
    print(form_data)
    df = pd.DataFrame(form_data, index=[0])
    df[['SplitLimit', 'CombinedSingleLimit']] = df['Policy_CombinedSingleLimit'].str.split('/', expand=True)
    df.drop("Policy_CombinedSingleLimit", axis=1, inplace=True)
    df.drop("Name", axis=1, inplace=True)
    dtype_dict = {
    "CustomerID": object,
    "InsuredAge": int,
    "InsuredZipCode": object,
    "InsuredGender": object,
    "InsuredEducationLevel": object,
    "InsuredOccupation": object,
    "InsuredHobbies": object,
    "CapitalGains": int,
    "CapitalLoss": int,
    "Country": object,
    "InsurancePolicyNumber": object,
    "CustomerLoyaltyPeriod": int,
    "DateOfPolicyCoverage": object,
    "InsurancePolicyState": object,
    "Policy_Deductible": int,
    "PolicyAnnualPremium": float,
    "UmbrellaLimit": int,
    "InsuredRelationship": object,
    "DateOfIncident": object,
    "TypeOfIncident": object,
    "TypeOfCollission": object,
    "SeverityOfIncident": object,
    "AuthoritiesContacted": object,
    "IncidentState": object,
    "IncidentCity": object,
    "IncidentAddress": object,
    "IncidentTime": float,
    "NumberOfVehicles": int,
    "PropertyDamage": object,
    "BodilyInjuries": int,
    "Witnesses": int,
    "PoliceReport": object,
    "AmountOfTotalClaim": float,
    "AmountOfInjuryClaim": int,
    "AmountOfPropertyClaim": int,
    "AmountOfVehicleDamage": int,
    "VehicleID": object,
    "VehicleMake": object,
    "VehicleModel": object,
    "VehicleYOM": object,
    "SplitLimit": int,
    "CombinedSingleLimit": int
    }
    df = df.astype(dtype_dict)
    # usning the pandas to_datetime() function we conver the columns into date-time format
    df['DateOfPolicyCoverage'] = pd.to_datetime(df['DateOfPolicyCoverage'], format='%Y-%m-%d')
    df['DateOfIncident'] = pd.to_datetime(df['DateOfIncident'], format='%Y-%m-%d')
    df['VehicleYOM'] = pd.to_datetime(df['VehicleYOM'], format='%Y')

    # function to convert columns of a dataframe into metioned data-type
    def convert_columns_types_to_category(DataFrame, cols=None, col_type = None):
        DataFrame[cols] = DataFrame[cols].astype(col_type) # Changing the Data Types using astype() function
        return DataFrame
    # getting categoric columns in ca_cols variable
    cat_cols = ['CustomerID', 'InsuredZipCode', 'InsuredGender', 'InsuredEducationLevel',
            'InsuredOccupation', 'InsuredHobbies', 'Country', 'InsurancePolicyNumber',
            'InsurancePolicyState', 'InsuredRelationship', 'TypeOfIncident', 'TypeOfCollission',
            'SeverityOfIncident', 'AuthoritiesContacted', 'IncidentState', 'IncidentCity',
            'IncidentAddress', 'PropertyDamage', 'PoliceReport', 'VehicleID',
            'VehicleMake', 'VehicleModel']
    # calling the convert_columns_types_to_category() function defined above
    df = convert_columns_types_to_category(df, cols=cat_cols, col_type = 'category')

    # Function to Drop Unccessary Columns in a Data Frame
    def drop_unnecessary_columns(DataFrame, cols=None):
        DataFrame.drop(columns = cols, axis = 1, inplace = True) # Dropping the columns
        return DataFrame

    # storing the columns which are to be dropped in drop_cols
    drop_cols = ['CustomerID', 'Country', 'InsurancePolicyNumber', 'VehicleID']

    # calling the function defined above to drop the columns
    df = drop_unnecessary_columns(df, cols=drop_cols)

    # Derive Age of Vehicle on the day of Incident
    df['VehicleAge'] = (df['DateOfIncident'] - df['VehicleYOM']).dt.days 

    # Derive Day of the Week of Incident
    df['DayOfWeek'] = df['DateOfIncident'].dt.day_name()

    # Derive Month of Incident
    df['MonthOfIncident'] = df['DateOfIncident'].dt.month_name()

    # Derive Time between Policy Coverage and Incident
    df['TimeBetweenCoverageAndIncident'] = (df['DateOfIncident'] - df['DateOfPolicyCoverage']).dt.days

    df.drop(['DateOfIncident', 'VehicleYOM', 'DateOfPolicyCoverage'], axis=1, inplace=True)

    # Change the dtype of 'DayOfWeek', 'MonthOfIncident' column from object to category
    df['DayOfWeek'] = df['DayOfWeek'].astype('category')
    df['MonthOfIncident'] = df['MonthOfIncident'].astype('category')

    InsuredZipCode_map = pickle.load(open('./Pickle Files/InsuredZipCode_map.pkl', 'rb'))
    InsuredGender_map = pickle.load(open('./Pickle Files/InsuredGender_map.pkl', 'rb'))
    InsuredEducationLevel_map = pickle.load(open('./Pickle Files/InsuredEducationLevel_map.pkl', 'rb'))
    InsuredOccupation_map = pickle.load(open('./Pickle Files/InsuredOccupation_map.pkl', 'rb'))
    InsuredHobbies_map = pickle.load(open('./Pickle Files/InsuredHobbies_map.pkl', 'rb'))
    InsurancePolicyState_map = pickle.load(open('./Pickle Files/InsurancePolicyState_map.pkl', 'rb'))
    InsuredRelationship_map = pickle.load(open('./Pickle Files/InsuredRelationship_map.pkl', 'rb'))
    TypeOfIncident_map = pickle.load(open('./Pickle Files/TypeOfIncident_map.pkl', 'rb'))
    TypeOfCollission_map = pickle.load(open('./Pickle Files/TypeOfCollission_map.pkl', 'rb'))
    SeverityOfIncident_map = pickle.load(open('./Pickle Files/SeverityOfIncident_map.pkl', 'rb'))
    AuthoritiesContacted_map = pickle.load(open('./Pickle Files/AuthoritiesContacted_map.pkl', 'rb'))
    IncidentState_map = pickle.load(open('./Pickle Files/IncidentState_map.pkl', 'rb'))
    IncidentCity_map = pickle.load(open('./Pickle Files/IncidentCity_map.pkl', 'rb'))
    IncidentAddress_map = pickle.load(open('./Pickle Files/IncidentAddress_map.pkl', 'rb'))
    PropertyDamage_map = pickle.load(open('./Pickle Files/PropertyDamage_map.pkl', 'rb'))
    PoliceReport_map = pickle.load(open('./Pickle Files/PoliceReport_map.pkl', 'rb'))
    VehicleMake_map = pickle.load(open('./Pickle Files/VehicleMake_map.pkl', 'rb'))
    VehicleModel_map = pickle.load(open('./Pickle Files/VehicleModel_map.pkl', 'rb'))
    DayOfWeek_map = pickle.load(open('./Pickle Files/DayOfWeek_map.pkl', 'rb'))
    MonthOfIncident_map = pickle.load(open('./Pickle Files/MonthOfIncident_map.pkl', 'rb'))

    df.loc[:,'InsuredZipCode'] = df['InsuredZipCode'].map(InsuredZipCode_map)
    df.loc[:,'InsuredGender'] = df['InsuredGender'].map(InsuredGender_map)
    df.loc[:,'InsuredEducationLevel'] = df['InsuredEducationLevel'].map(InsuredEducationLevel_map)
    df.loc[:,'InsuredOccupation'] = df['InsuredOccupation'].map(InsuredOccupation_map)
    df.loc[:,'InsuredHobbies'] = df['InsuredHobbies'].map(InsuredHobbies_map)
    df.loc[:,'InsurancePolicyState'] = df['InsurancePolicyState'].map(InsurancePolicyState_map)
    df.loc[:,'InsuredRelationship'] = df['InsuredRelationship'].map(InsuredRelationship_map)
    df.loc[:,'TypeOfIncident'] = df['TypeOfIncident'].map(TypeOfIncident_map)
    df.loc[:,'TypeOfCollission'] = df['TypeOfCollission'].map(TypeOfCollission_map)
    df.loc[:,'SeverityOfIncident'] = df['SeverityOfIncident'].map(SeverityOfIncident_map)
    df.loc[:,'AuthoritiesContacted'] = df['AuthoritiesContacted'].map(AuthoritiesContacted_map)
    df.loc[:,'IncidentState'] = df['IncidentState'].map(IncidentState_map)
    df.loc[:,'IncidentCity'] = df['IncidentCity'].map(IncidentCity_map)
    df.loc[:,'IncidentAddress'] = df['IncidentAddress'].map(IncidentAddress_map)
    df.loc[:,'PropertyDamage'] = df['PropertyDamage'].map(PropertyDamage_map)
    df.loc[:,'PoliceReport'] = df['PoliceReport'].map(PoliceReport_map)
    df.loc[:,'VehicleMake'] = df['VehicleMake'].map(VehicleMake_map)
    df.loc[:,'VehicleModel'] = df['VehicleModel'].map(VehicleModel_map)
    df.loc[:,'DayOfWeek'] = df['DayOfWeek'].map(DayOfWeek_map)
    df.loc[:,'MonthOfIncident'] = df['MonthOfIncident'].map(MonthOfIncident_map)

    reverse_InsuredZipCode_map = {v: k for k, v in InsuredZipCode_map.items()}
    reverse_InsuredGender_map = {v: k for k, v in InsuredGender_map.items()}
    reverse_InsuredEducationLevel_map = {v: k for k, v in InsuredEducationLevel_map.items()}
    reverse_InsuredOccupation_map = {v: k for k, v in InsuredOccupation_map.items()}
    reverse_InsuredHobbies_map = {v: k for k, v in InsuredHobbies_map.items()}
    reverse_InsurancePolicyState_map = {v: k for k, v in InsurancePolicyState_map.items()}
    reverse_InsuredRelationship_map = {v: k for k, v in InsuredRelationship_map.items()}
    reverse_TypeOfIncident_map = {v: k for k, v in TypeOfIncident_map.items()}
    reverse_TypeOfCollission_map = {v: k for k, v in TypeOfCollission_map.items()}
    reverse_SeverityOfIncident_map = {v: k for k, v in SeverityOfIncident_map.items()}
    reverse_AuthoritiesContacted_map = {v: k for k, v in AuthoritiesContacted_map.items()}
    reverse_IncidentState_map = {v: k for k, v in IncidentState_map.items()}
    reverse_IncidentCity_map = {v: k for k, v in IncidentCity_map.items()}
    reverse_IncidentAddress_map = {v: k for k, v in IncidentAddress_map.items()}
    reverse_PropertyDamage_map = {v: k for k, v in PropertyDamage_map.items()}
    reverse_PoliceReport_map = {v: k for k, v in PoliceReport_map.items()}
    reverse_VehicleMake_map = {v: k for k, v in VehicleMake_map.items()}
    reverse_VehicleModel_map = {v: k for k, v in VehicleModel_map.items()}
    reverse_DayOfWeek_map = {v: k for k, v in DayOfWeek_map.items()}
    reverse_MonthOfIncident_map = {v: k for k, v in MonthOfIncident_map.items()}

    imputer = pickle.load(open('./Pickle Files/Missing_Value_KNN_Imputer.pkl', 'rb'))
    cols = df.columns
    print(cols)
    imputed_array = imputer.transform(df[cols])
    df_imp = pd.DataFrame(imputed_array, columns = cols)

    df_imp.loc[:,'InsuredZipCode'] = df_imp['InsuredZipCode'].map(reverse_InsuredZipCode_map)
    df_imp.loc[:,'InsuredGender'] = df_imp['InsuredGender'].map(reverse_InsuredGender_map)
    df_imp.loc[:,'InsuredEducationLevel'] = df_imp['InsuredEducationLevel'].map(reverse_InsuredEducationLevel_map)
    df_imp.loc[:,'InsuredOccupation'] = df_imp['InsuredOccupation'].map(reverse_InsuredOccupation_map)
    df_imp.loc[:,'InsuredHobbies'] = df_imp['InsuredHobbies'].map(reverse_InsuredHobbies_map)
    df_imp.loc[:,'InsurancePolicyState'] = df_imp['InsurancePolicyState'].map(reverse_InsurancePolicyState_map)
    df_imp.loc[:,'InsuredRelationship'] = df_imp['InsuredRelationship'].map(reverse_InsuredRelationship_map)
    df_imp.loc[:,'TypeOfIncident'] = df_imp['TypeOfIncident'].map(reverse_TypeOfIncident_map)
    df_imp.loc[:,'TypeOfCollission'] = df_imp['TypeOfCollission'].map(reverse_TypeOfCollission_map)
    df_imp.loc[:,'SeverityOfIncident'] = df_imp['SeverityOfIncident'].map(reverse_SeverityOfIncident_map)
    df_imp.loc[:,'AuthoritiesContacted'] = df_imp['AuthoritiesContacted'].map(reverse_AuthoritiesContacted_map)
    df_imp.loc[:,'IncidentState'] = df_imp['IncidentState'].map(reverse_IncidentState_map)
    df_imp.loc[:,'IncidentCity'] = df_imp['IncidentCity'].map(reverse_IncidentCity_map)
    df_imp.loc[:,'IncidentAddress'] = df_imp['IncidentAddress'].map(reverse_IncidentAddress_map)
    df_imp.loc[:,'PropertyDamage'] = df_imp['PropertyDamage'].map(reverse_PropertyDamage_map)
    df_imp.loc[:,'PoliceReport'] = df_imp['PoliceReport'].map(reverse_PoliceReport_map)
    df_imp.loc[:,'VehicleMake'] = df_imp['VehicleMake'].map(reverse_VehicleMake_map)
    df_imp.loc[:,'VehicleModel'] = df_imp['VehicleModel'].map(reverse_VehicleModel_map)
    df_imp.loc[:,'DayOfWeek'] = df_imp['DayOfWeek'].map(reverse_DayOfWeek_map)
    df_imp.loc[:,'MonthOfIncident'] = df_imp['MonthOfIncident'].map(reverse_MonthOfIncident_map)

    # getting categoric columns in cat_cols variable
    cat_cols = ['InsuredZipCode', 'InsuredGender', 'InsuredEducationLevel', 'InsuredOccupation',
                'InsuredHobbies', 'InsurancePolicyState', 'InsuredRelationship', 'TypeOfIncident',
                'TypeOfCollission', 'SeverityOfIncident', 'AuthoritiesContacted', 'IncidentState',
                'IncidentCity', 'IncidentAddress', 'PropertyDamage', 'PoliceReport', 
                'VehicleMake', 'VehicleModel', 'DayOfWeek', 'MonthOfIncident']

    # calling the convert_columns_types_to_category() function defined above
    test_df_imp = convert_columns_types_to_category(df_imp, cols=cat_cols, col_type = 'category')

    def get_num_cat_dataframes(DataFrame):
        num_df = DataFrame.select_dtypes(include=['int','float']) # Assigning the columns which are of 'int' & 'float' type to num_df
        cat_df = DataFrame.select_dtypes(include=['category']) # Assigning the columns which are of 'category' type to cat_df
        return num_df, cat_df
    df_num, df_cat = get_num_cat_dataframes(df_imp)



    scaler = pickle.load(open('./Pickle Files/StandardScaler.pkl', 'rb'))
    def perform_standardization_one_df(test_df, scaler):
        num_attr = test_df.select_dtypes(['int','float']).columns
        test_df[num_attr]=scaler.transform(test_df[num_attr])
    perform_standardization_one_df(df_num, scaler)

    InsuredGender_ohe = pickle.load(open('./Pickle Files/InsuredGender_ohe.pkl', 'rb'))        
    InsurancePolicyState_ohe = pickle.load(open('./Pickle Files/InsurancePolicyState_ohe.pkl', 'rb'))        
    TypeOfIncident_ohe = pickle.load(open('./Pickle Files/TypeOfIncident_ohe.pkl', 'rb'))        
    TypeOfCollission_ohe = pickle.load(open('./Pickle Files/TypeOfCollission_ohe.pkl', 'rb'))        
    AuthoritiesContacted_ohe = pickle.load(open('./Pickle Files/AuthoritiesContacted_ohe.pkl', 'rb'))        
    IncidentState_ohe = pickle.load(open('./Pickle Files/IncidentState_ohe.pkl', 'rb'))        
    IncidentCity_ohe = pickle.load(open('./Pickle Files/IncidentCity_ohe.pkl', 'rb'))        
    PropertyDamage_ohe = pickle.load(open('./Pickle Files/PropertyDamage_ohe.pkl', 'rb'))        
    PoliceReport_ohe = pickle.load(open('./Pickle Files/PoliceReport_ohe.pkl', 'rb'))        
    DayOfWeek_ohe = pickle.load(open('./Pickle Files/DayOfWeek_ohe.pkl', 'rb'))        
    MonthOfIncident_ohe = pickle.load(open('./Pickle Files/MonthOfIncident_ohe.pkl', 'rb'))

    def perform_one_hot_encoding_one_df(X_test,cols,ohe, prefix):
        X_test_new = ohe.transform(X_test[cols])
        for i in range(len(ohe.categories_)):
            if i == 0:
                label = ohe.categories_[i]
                label_combined = label[1:]
                label_combined = [prefix + label for label in label_combined]
            elif i > 0:
                label = ohe.categories_[i]
                label_combined = np.append(label_combined,label[1:])
                label_combined = [prefix + label for label in label_combined]
        # Adding Transformed X_test back to the main DataFrame
        X_test[label_combined] = X_test_new
    # Dropping the Encoded Column
        X_test.drop(cols,axis=1,inplace=True)
        return None
    col=['InsuredGender']
    perform_one_hot_encoding_one_df(df_cat, col, InsuredGender_ohe, prefix='InsuredGender_')
    col=['InsurancePolicyState']
    perform_one_hot_encoding_one_df(df_cat, col, InsurancePolicyState_ohe, prefix='InsurancePolicyState_')
    col=['TypeOfIncident']
    perform_one_hot_encoding_one_df(df_cat, col, TypeOfIncident_ohe, prefix='TypeOfIncident_')
    col=['TypeOfCollission']
    perform_one_hot_encoding_one_df(df_cat, col, TypeOfCollission_ohe, prefix='TypeOfCollission_')
    col=['AuthoritiesContacted']
    perform_one_hot_encoding_one_df(df_cat, col, AuthoritiesContacted_ohe, prefix='AuthoritiesContacted_')
    col=['IncidentState']
    perform_one_hot_encoding_one_df(df_cat, col, IncidentState_ohe, prefix='IncidentState_')
    col=['IncidentCity']
    perform_one_hot_encoding_one_df(df_cat, col, IncidentCity_ohe, prefix='IncidentCity_')
    col=['PropertyDamage']
    perform_one_hot_encoding_one_df(df_cat, col, PropertyDamage_ohe, prefix='PropertyDamage_')
    col=['PoliceReport']
    perform_one_hot_encoding_one_df(df_cat, col, PoliceReport_ohe, prefix='PoliceReport_')
    col=['DayOfWeek']
    perform_one_hot_encoding_one_df(df_cat, col, DayOfWeek_ohe, prefix='DayOfWeek_')
    col=['MonthOfIncident']
    perform_one_hot_encoding_one_df(df_cat, col, MonthOfIncident_ohe, prefix='MonthOfIncident_')

    InsuredEducationLevel_encoder = pickle.load(open('./Pickle Files/InsuredEducationLevel_encoder.pkl', 'rb'))
    InsuredRelationship_encoder = pickle.load(open('./Pickle Files/InsuredRelationship_encoder.pkl', 'rb'))
    SeverityOfIncident_encoder = pickle.load(open('./Pickle Files/SeverityOfIncident_encoder.pkl', 'rb'))
    

    def ordinal_encoder_one_df(X_test, column_name, encoder):
        # Transform the categorical data in X_test using the encoder
        X_test_encoded = encoder.transform(X_test[[column_name]])
        # Replace the original categorical column in X_train and X_test with the encoded values
        X_test[column_name] = X_test_encoded
        return None
    ordinal_encoder_one_df(df_cat, 'InsuredEducationLevel', InsuredEducationLevel_encoder)
    ordinal_encoder_one_df(df_cat, 'InsuredRelationship', InsuredRelationship_encoder)
    ordinal_encoder_one_df(df_cat, 'SeverityOfIncident', SeverityOfIncident_encoder)

    target_encoder = pickle.load(open('./Pickle Files/target_encoder.pkl', 'rb'))
    df_cat_encoded = target_encoder.transform(df_cat)

    df_num.reset_index(inplace = True, drop = True)
    df_cat_encoded.reset_index(inplace = True, drop = True)

    # Combine numerical data and categorical data
    def combine_num_df_cat_df(num_df, cat_df):
        result = pd.concat([num_df,cat_df],axis=1) # Using concat funtion in pandas we join the numerical columns and categorical columns
        return result
    df = combine_num_df_cat_df(df_num, df_cat_encoded)

    model = pickle.load(open('./Pickle Files/svc_gscv_model.pkl', 'rb'))

    # Predict on the train set
    pred = model.predict(df)

    print(pred[0])
    if pred[0] == 0:
        return render_template("prediction-result.html", prediction_result=0, CustomerID = CustomerID, Name = Name)
    elif pred[0] == 1:
        return render_template("prediction-result.html", prediction_result=1, CustomerID = CustomerID, Name = Name)

@app.route('/overview')
def steps():
    return render_template('steps.html')

@app.route('/visualizations')
def visualizations():
    return render_template('visuals.html')

if __name__=="__main__":
    app.run(debug = True)