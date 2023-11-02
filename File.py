
import pandas as pd
import streamlit as st
import pickle 
from sklearn.preprocessing import LabelEncoder
 


def main():
    html_temp="""
    <div style ="background-color:Turquoise;padding:16px">
    
    <h2 style="color:black;text-align:center;"> Car Price Prediction Using ML </h2>
    </div>
    """
    df= pd.read_csv("https://drive.google.com/uc?export=download&id=17KxRuOK4uONRZyfYNgjoX62y2yO5KpLY")
    
    lb= LabelEncoder()
    
    with open('best_model.pkl','rb') as test_file:
       loaded_model= pickle.load(test_file)
    
 
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    st.write(' ')
    st.write(' ')
    
    st.markdown("##### Want To Sell Your Car?\n##### Let's see what is your demandeble price")
    r1 = st.selectbox(" Car's Brand", df['name'].unique())
    
    r2=st.number_input("Travelled distance in Kilometer",100,500000,step=100)
    
    r3=st.selectbox("Type of Fuel",df['fuel'].unique())
       
    r4=st.selectbox("Seller Category",df['seller_type'].unique())

    r5=st.selectbox("Car's Charecter",df['transmission'].unique())
    
    r6=st.selectbox("Numbers of pre-owner's",df['owner'].unique())
    
    r7=st.slider(" Purchase Year",1990,2023)
    
    
    
    data_new=pd.DataFrame({
        'name': r1,
        'km_driven':r2,
        'fuel':r3,
        'seller_type':r4,
        'transmission':r5,
        'owner':r6,
        'age':r7
        
    },index=[0])
    
    data_new['name']=lb.fit_transform(data_new['name'])
    data_new['fuel']=lb.fit_transform(data_new['fuel'])
    data_new['seller_type']=lb.fit_transform(data_new['seller_type'])
    data_new['transmission']=lb.fit_transform(data_new['transmission'])
    data_new['owner']=lb.fit_transform(data_new['owner'])

        
           
    if st.button('Predict'):
        pred=loaded_model.predict(data_new)
        
        st.balloons()
        message="You can sell your car for {:.2f} lakhs".format(pred[0])
        
        st.success(message)
    
        
        
    
    
  
    

     
    

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
if __name__ == '__main__':
	main()

