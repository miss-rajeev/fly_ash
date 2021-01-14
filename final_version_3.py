
#Can we make a compact dashboard across several columns and with a dark theme?"""
import io
from typing import List, Optional

#import markdown
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly import express as px
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
from pulp import *
from itertools import product
import time
import requests, json
from geopandas import GeoDataFrame
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from plotly.subplots import make_subplots
from io import BytesIO
import base64
from typing import List, Optional
from math import *


st.beta_set_page_config(layout="wide")
#st.title("Fly Ash Procurement")
st.markdown("<h1 style='text-align: center; color: red;'>Fly Ash Procurement Dashboard</h1>", unsafe_allow_html=True)
side_bg = "ABG.jpg"
side_bg_ext = "jpg"

st.markdown(""" <style> body {color:#2F4F4F  ;background-color: #e9e9e9 ; } </style> """, unsafe_allow_html=True)



#bg

# matplotlib.use("TkAgg")
matplotlib.use("Agg")
COLOR = "black"
BACKGROUND_COLOR = "#fff"
@st.cache
def load_image(img):
    im =Image.open(os.path.join(img))
    return im

@st.cache
def load_images_gdna():
    b=load_image("gdnaf2.PNG")
    return b 

def clear_linkagesp(p):
    st.markdown(
            f"""<style>
            .main .block-container div > .element-container:nth-child(-n+{p})  {{
            width:1220px !important;display:none;}}          
            </style>""",
            unsafe_allow_html=True)
#def sidebar_linkages(p):
#    st.markdown(f"""<style>)
#
##"""
#functions to upload xlsx files and download it
#"""
def to_excel(source_capacity,plant_demand,basic_cost,freight_cost,lat_long):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    #data.to_excel(writer,sheet_name='Data',index=None)
    source_capacity.to_excel(writer, sheet_name='source_capacity',index=None)
    plant_demand.to_excel(writer,sheet_name='plant_demand',index=None)
    basic_cost.to_excel(writer,sheet_name='basic_cost',index=None)
    freight_cost.to_excel(writer,sheet_name='freight_cost',index=None)
    lat_long.to_excel(writer,sheet_name='lat_long',index=None)
    #ptpkm.to_excel(writer,sheet_name='PTPKM',index=None)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(a,b,c,d,e,f):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(a,b,c,d,e,f)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download input file</a>' # decode b'abc' => abc

@st.cache              
def retro_dictify(frame):       #create dictionary
    d = {}
    for row in frame.values:
        here = d
        for elem in row[:-2]:
            if elem not in here:
                here[elem] = {}
            here = here[elem]
        here[row[-2]] = row[-1]
    return d

def data1_prep(data,df_input,df_ptpkm_dis):
    data1=data.copy()
    data1=pd.merge(df_input,df_ptpkm_dis[['FromID','ToID','ptpkm','Kilometres']],how='left',left_on=['Source_code','Cluster_code'],right_on=['FromID','ToID'])
    data1['ptpkm_y']=np.where(data1['ptpkm_y'].isnull(),data1['ptpkm_x'],data1['ptpkm_y'])
    data1['Kilometres']=np.where(data1['Kilometres'].isnull(),data1['freight_new']/data1['ptpkm_x'],data1['Kilometres'])
    data1['Kilometres']=data1['Kilometres'].fillna(0)
    data1=data1.rename(columns={'ptpkm_y':'ptpkm'})
    data1['ptpkm']=np.where(data1['ptpkm']==inf, 0, data1['ptpkm'])
    data1['Total_cost']=((((data1['Kilometres']*data1['ptpkm'])+data1['Other(Rs/MT)'])+data1['Basic Rate(Rs/MT)'])*data1['Qty(LE)(LMT)'])
    return data1

@st.cache(allow_output_mutation=True)
def fileUpload(df,sheet) -> pd.DataFrame:
    return pd.read_excel(df,sheet)

@st.cache(allow_output_mutation=True)
def fileUpload_csv(df) -> pd.DataFrame:
    return pd.read_csv(df)


def to_excel_result(source_capacity,plant_demand,basic_cost):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    source_capacity.to_excel(writer, sheet_name='Summary',index=None)
    plant_demand.to_excel(writer,sheet_name='Reallocation',index=None)
    basic_cost.to_excel(writer,sheet_name='Existing',index=None)
    
    
    writer.save()
    processed_data = output.getvalue()
    return processed_data

    #
    #"""
    #1) input file
    #2) source capacity, plant demand, ptpkm, handling and basic cost
    #"""

@st.cache    
def line_plot1(df_linkages,df1,df2,s1,s2,c,n1,n2,c1,c2,nl1):
    df_linkages=df_linkages.drop_duplicates()
    df_linkages.index=range(0,len(df_linkages))
    fig = go.Figure(go.Scattermapbox(
        mode = "markers",
        lon = df1.iloc[:,2],
        lat = df1.iloc[:,1],
        marker = {'size': s1,'color': c1},
        text=df1.iloc[:,0],
        hoverinfo='text',
        name=n1))
   
    fig.add_trace(go.Scattermapbox(
       mode = "markers",
       lon = df2.iloc[:,2],
       lat = df2.iloc[:,1],
       marker = {'size': s2,'color':c2},
       text=df2.iloc[:,0],
       hoverinfo='text',
       name=n2))
    for i in range(0,1):    #x1y1, x2y2 link
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[df_linkages.iloc[:,2][i],df_linkages.iloc[:,5][i]],  #y1y2
           lat=[df_linkages.iloc[:,1][i],df_linkages.iloc[:,4][i]],   #x1x2
            line=dict(width = 1, color = c),
            opacity=0.7,
            text=df_linkages.iloc[:,0][i]+'-'+df_linkages.iloc[:,3][i],
            hoverinfo='text',
            below=None,
            name=nl1,
            showlegend=True))

    for i in range(1,len(df_linkages)):
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[df_linkages.iloc[:,2][i],df_linkages.iloc[:,5][i]],
           lat=[df_linkages.iloc[:,1][i],df_linkages.iloc[:,4][i]],
            line=dict(width = 1, color = c),
            opacity=0.7,
            text=df_linkages.iloc[:,0][i]+'-'+df_linkages.iloc[:,3][i],
            hoverinfo='text',
            below=None,
            showlegend=False))
    
   
    fig.update_layout(
       margin ={'l':0,'t':0,'b':0,'r':0},
       mapbox = {
           'center': {'lon': 78.9629, 'lat': 20.5937},
           'style': "stamen-terrain",
           'center': {'lon': 78.9629, 'lat': 20.5937},
           'zoom': 3.5},legend=dict(x=1,y=.1)
           )
    return fig

@st.cache
def line_plot2(df_linkages,df1_linkages,df1,df2,s1,s2,c,n1,n2,c1,c2,cc,nl1,nl2,df1e,df2e):
    df_linkages=df_linkages.drop_duplicates()
    df_linkages.index=range(0,len(df_linkages))
    fig = go.Figure(go.Scattermapbox(
        mode = "markers",
        lon = df1.iloc[:,2],
        lat = df1.iloc[:,1],
        marker = {'size': s1,'color': c1},
        text=df1.iloc[:,0],
        hoverinfo='text',
        name=n1))
   
    fig.add_trace(go.Scattermapbox(
       mode = "markers",
       lon = df2.iloc[:,2],
       lat = df2.iloc[:,1],
       marker = {'size': s2,'color':c2},
       text=df2.iloc[:,0],
       hoverinfo='text',
       name=n2))
    
    fig.add_trace(go.Scattermapbox(
       mode = "markers",
       lon = df1e.iloc[:,2],
       lat = df1e.iloc[:,1],
       marker = {'size': s1,'color':c1},
       text=df1e.iloc[:,0],
       hoverinfo='text',
       name=n1,
       showlegend=False))
    fig.add_trace(go.Scattermapbox(
       mode = "markers",
       lon = df2e.iloc[:,2],
       lat = df2e.iloc[:,1],
       marker = {'size': s2,'color':c2},
       text=df2e.iloc[:,0],
       hoverinfo='text',
       name=n2,
       showlegend=False))
    for i in range(0,1):
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[df_linkages.iloc[:,2][i],df_linkages.iloc[:,5][i]],
            lat=[df_linkages.iloc[:,1][i],df_linkages.iloc[:,4][i]],
            line=dict(width = 1, color = c),
            opacity=1,
            text=df_linkages.iloc[:,0][i]+'-'+df_linkages.iloc[:,3][i],
            hoverinfo='text',
            below=None,
            name=nl1,
            showlegend=True))
    for i in range(0,1):
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[df1_linkages.iloc[:,2][i],df1_linkages.iloc[:,5][i]],
           lat=[df1_linkages.iloc[:,1][i],df1_linkages.iloc[:,4][i]],
            line=dict(width = 1, color = cc),
            opacity=1,
            text=df1_linkages.iloc[:,0][i]+'-'+df1_linkages.iloc[:,3][i],
            hoverinfo='text',
            below=None,
            name=nl2,
            showlegend=True))

    for i in range(0,len(df_linkages)):
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[df_linkages.iloc[:,2][i],df_linkages.iloc[:,5][i]],
           lat=[df_linkages.iloc[:,1][i],df_linkages.iloc[:,4][i]],
            line=dict(width = 1, color = c),
            opacity=1,
            text=df_linkages.iloc[:,0][i]+'-'+df_linkages.iloc[:,3][i],
            hoverinfo='text',
            below=None,
            showlegend=False))
    
    for i in range(0,len(df1_linkages)):
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[df1_linkages.iloc[:,2][i],df1_linkages.iloc[:,5][i]],
           lat=[df1_linkages.iloc[:,1][i],df1_linkages.iloc[:,4][i]],
            line=dict(width = 1, color = cc),
            opacity=1,
            text=df1_linkages.iloc[:,0][i]+'-'+df1_linkages.iloc[:,3][i],
            hoverinfo='text',
            below=None,
            showlegend=False))
    
   
    fig.update_layout(
       margin ={'l':0,'t':0,'b':0,'r':0},
       mapbox = {
           'center': {'lon': 78.9629, 'lat': 20.5937},
           'style': "stamen-terrain",
           'center': {'lon': 78.9629, 'lat': 20.5937},
           'zoom': 3.5},legend=dict(x=1,y=.1)
           )
    return fig



def main():
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.markdown(
    f"""
    <style>
       .sidebar .sidebar-content {{
        background:#C0C0C0 no-repeat url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg , "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True)
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
#    st.markdown("<h2 style='text-align: center; color: black;'>This Dashboard can be used to optimise Fly Ash Procurement costs for Ultra Tech Cements. </h2>", unsafe_allow_html=True)
    st.markdown("<s2 style='text-align: center; color: black;'>This Dashboard can be used to optimise Fly Ash Procurement costs for Ultra Tech Cements. The modules available in the dashboard are : </s2>", unsafe_allow_html=True)
#    st.subheader("This Dashboard can be used to optimise Fly Ash Procurement costs for Ultra Tech Cements. The modules available in the dashboard are : ")
    st.markdown("<h4 style='text-align: left; color: black;'>1. Data Viewer : </h4>", unsafe_allow_html=True)
    st.markdown("This page enables you to view the existing data and also to update the input file via a file uploader. ")
    st.markdown("<h4 style='text-align: left; color: black;'>2. Map Viewer : </h4>", unsafe_allow_html=True)
    st.markdown("This page enables you to view the existing fly ash linkages in a map.")
    st.markdown("<h4 style='text-align: left; color: black;'>3. Update Input : </h4>", unsafe_allow_html=True)
    st.markdown("This page enables you to update plant demand and source capacity  ")
    st.markdown("<h4 style='text-align: left; color: black;'>4. Optimiser : </h4>", unsafe_allow_html=True)
    st.markdown("This page runs the optimiser and enables you to download the optimised results")
    st.markdown("<h4 style='text-align: left; color: black;'>5. Results Summary : </h4>", unsafe_allow_html=True)
    st.markdown("This enables you to view the results in a brief format.")
#    st.write("This is a dashboard for optimising fly ash linkages")
    st.markdown('---')
    st.text("Please click on the data viewer present on the sidebar to your left to proceed")
    if st.sidebar.checkbox("Data Viewer"):
        clear_linkagesp(17)
        st.markdown("<h1 style='text-align: center; color: red;'>Fly Ash Procurement Dashboard</h1>", unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; color:black  ;'>Data Viewer</h2>", unsafe_allow_html=True)
 
        #input data
        path_input='final_input_file1.xlsx'
        plant_demand=fileUpload(path_input,'Demand_new')
        source_capacity=fileUpload(path_input,'Capacity_new')
        freight_cost=fileUpload(path_input,'Freight_cost')
        basic_cost=fileUpload(path_input,'Basic_cost')
        key_plant=fileUpload(path_input,'key_plant')
        key_source=fileUpload(path_input,'key_source')
        lat_long=fileUpload(path_input,'LatLong_name_mapping')
        lat_long['FromID']=lat_long['FromID'].astype(str)        
        df_input=fileUpload(path_input,'Existing Data')
        df_input=df_input.drop_duplicates()
        df_input=df_input[df_input["Cluster/Unit"].isin(key_plant.Plant.unique().tolist())]
        df_input2=df_input[~(df_input['Source_code'].isnull())]
        df_input['grade_new']=df_input['material_group_new']+'='+df_input['category']
        
        #source capacity
#        source_capacity_ex=pd.read_excel('input_files.xlsx','Source_capacity')
#        source_capacity_ex['grade_new']=source_capacity_ex['material_group_new']+'='+source_capacity_ex['category']
        #plant demand
#        plant_demand_ex=pd.read_excel('input_files.xlsx','plant_demand')
#        plant_demand_ex['grade_new']=plant_demand_ex['material_group_new']+'='+plant_demand_ex['category']
        #basic cost
        basic_cost_ex=fileUpload('input_files.xlsx','basic_cost')
        
        basic_cost_ex['grade_new']=basic_cost_ex['material_group_new']+'='+basic_cost_ex['category']
        #handling cost
        handling_cost=fileUpload('input_files.xlsx','handling_cost')
        handling_cost['grade_new']=handling_cost['material_group_new']+'='+handling_cost['category']
        #ptpkm
        ptpkm=fileUpload('input_files.xlsx','PTPKM')     
#        df_input=df_input["Source Name"].isin(key_source.Name.unique().tolist()) 
        #distance file
        df_dis=fileUpload_csv(r'distance_final1.csv') 
        file_lat_long=df_dis[['FromID','FromLatitude', 'FromLongitude']].copy()
        file_lat_long=file_lat_long.drop_duplicates(['FromID'])
  #    file_lat_long_groupby=file_lat_long.groupby(['FromID']).count().reset_index()
        data=df_input[['source_name_given','Cluster/Unit','material_group_new','category','grade_new','Qty(LE)(LMT)']]
        data_map=df_input2.copy()
        data_map=pd.merge(data_map,file_lat_long,how='left',left_on=['Source_code'],right_on=['FromID'])
        data_map=data_map.rename(columns={'FromLatitude':'Source_latitude','FromLongitude':'Source_longitude'})
        del data_map['FromID']
        data_map=pd.merge(data_map,file_lat_long,how='left',left_on=['Cluster_code'],right_on=['FromID'])
        data_map=data_map.rename(columns={'FromLatitude':'Cluster_latitude','FromLongitude':'Cluster_longiutde'})
        del data_map['FromID']
        
        file_view=st.selectbox("Select appropriate option to view data", ["Demand","Capacity","Basic cost","Freight cost","Lat Long"])
        if file_view == "Demand":
            plant_demand.index=range(1,len(plant_demand)+1)
            st.write(plant_demand)
            plant_demand.index=range(0,len(plant_demand))
        elif file_view == "Capacity":
            source_capacity.index=range(1,len(source_capacity)+1)
            st.write(source_capacity)
            source_capacity.index=range(0,len(source_capacity))
        elif file_view == "Basic cost":
            basic_cost.index=range(1,len(basic_cost)+1)
            st.write(basic_cost)
            basic_cost.index=range(0,len(basic_cost))
        elif file_view == "Freight cost":
            freight_cost.index=range(1,len(freight_cost)+1)
            st.write(freight_cost)
            freight_cost.index=range(0,len(freight_cost))     
        elif file_view=="Lat Long":
            lat_long.index=range(1,len(lat_long)+1)
            st.write(lat_long)
            lat_long.index=range(0,len(lat_long)) 
        else:
            st.write("")
            
        
        val = to_excel(source_capacity,plant_demand,basic_cost,freight_cost,lat_long)
        b64 = base64.b64encode(val)
        href1= f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Click to download the input file format </a>' # decode b'abc' => abc
        st.markdown(href1, unsafe_allow_html=True)
        #st.write('')
        st.markdown("<body style='color: grey;'><p>(Format to be used while uploading a new file)</p></body>", unsafe_allow_html=True)
        st.subheader('Note for user')
        st.write(
            " 1) Please use the update option below to make changes  \n"
            " 2) For large number of changes, it is advisable to upload a new file in the required format  \n"
        )   
        
        
        uploader_prim=st.file_uploader("Update file (in appropriate format)", type="xlsx")
        st.text("Please click on the map viewer present on the sidebar to your left to proceed")
        if uploader_prim is not None:
            df_dict=pd.read_excel(uploader_prim, None)
            source_capacity=df_dict['Source_capacity']
#            source_capacity['grade_new']=source_capacity['Grade']+'='+source_capacity['Category']
            plant_demand=df_dict['plant_demand']
#            plant_demand['grade_new']=plant_demand['Grade']+'='+plant_demand['Category']
            basic_cost=df_dict['basic_cost']
#            basic_cost['grade_new']=basic_cost['Grade']+'='+basic_cost['Category']
            freight_cost=df_dict['freight_cost']
#            freight_cost['grade_new']=freight_cost['Grade']+'='+freight_cost['Category']
            lat_long=df_dict['lat_long']
            
        if st.sidebar.checkbox('Map Viewer'):
            clear_linkagesp(40)
            st.markdown("<h1 style='text-align: center; color: red;'>Fly Ash Procurement Dashboard</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color:black  ;'>Map Viewer</h2>", unsafe_allow_html=True)
            #columns for filter
            st.subheader("Select the appropriate filters")
            #col1, col2 = st.beta_columns(2)
            #col1=st.beta_columns(3)
            
            fly_ash_type=st.multiselect("Select Fly ash type",data['material_group_new'].unique().tolist())
            if len(fly_ash_type)!=0:
                data1=data[data['material_group_new'].isin(fly_ash_type)]
                data_map1=data_map[data_map['material_group_new'].isin(fly_ash_type)]
            else:
                data1=data.copy()
                data_map1=data_map.copy()
            
            source_list=st.multiselect("Select sources",data1['source_name_given'].unique().tolist())
            if len(source_list)!=0:
                data1=data1[data1['source_name_given'].isin(source_list)]
                data_map1=data_map1[data_map1['source_name_given'].isin(source_list)]
            else:
                data1=data1.copy()
                data_map1=data_map1.copy()
            plant_list=st.multiselect("Select plant",data1['Cluster/Unit'].unique().tolist())
            if len(plant_list)!=0:
                data1=data1[data1['Cluster/Unit'].isin(plant_list)]
                data_map1=data_map1[data_map1['Cluster/Unit'].isin(plant_list)]
            else:
                data1=data1.copy()
                data_map1=data_map1.copy()
            category_list=st.multiselect("Select category",data1['category'].unique().tolist())
            if len(category_list)!=0:
                data1=data1[data1['category'].isin(category_list)]
                data_map1=data_map1[data_map1['category'].isin(category_list)]
            else:
                data1=data1.copy()
                data_map1=data_map1.copy()
            
            #st.write(data1)
            #st.markdown('---')
            st.markdown("<h2 style='text-align: justify; color: black;'>Existing Map View</h2>", unsafe_allow_html=True)
    #        data1.index=range(1,len(data1)+1)
    #        st.write(data1)
    #        data1.index=range(0,len(data1))
    #        val = to_excel(source_capacity,plant_demand,basic_cost,handling_cost,ptpkm,data)
    #        b64 = base64.b64encode(val)
    #        href1= f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download existing file</a>' # decode b'abc' => abc
    #        st.write(" ")
    #        st.markdown(href1, unsafe_allow_html=True)
    #        st.markdown('---')
            
            
            
            
            data_map2=data_map1[['source_name_given','Source_latitude','Source_longitude','Cluster/Unit','Cluster_latitude','Cluster_longiutde','Qty(LE)(LMT)']]
            data_map2=data_map2.drop_duplicates()
            df_s=data_map2[['source_name_given','Source_latitude','Source_longitude']].copy()
            df_s=df_s.drop_duplicates()
            df_p=data_map2[['Cluster/Unit','Cluster_latitude','Cluster_longiutde']].copy()
            df_p=df_p.drop_duplicates()
    #        s1=7
    #        s2=9
    #        c='blue'
    #        n1='Sources'
    #        n2='Plants'
    #        c1='red'
    #        c2='green'
    #        cc='brown'
    #        nl1="Existing"
    #        nl2="Optimal"
            s1=7
            s2=9
            c='blue'
            n1='Sources'
            n2='Plants'
            c1='red'
            c2='green'
            nl1="Existing"
            fig_l2=line_plot1(data_map2,df_s,df_p,s1,s2,c,n1,n2,c1,c2,nl1)
            #map_layer = st.beta_columns(5)
           
            #map_layer[1].plotly_chart(fig_l2)
            st.plotly_chart(fig_l2)
            st.text("Please click on the Update Input present on the sidebar to your left to proceed")
            
            if st.sidebar.checkbox("Update Input"):
                plant_demand[['Grade', 'Category']]=plant_demand['grade_new'].str.split("=",expand=True)
                try:
                    plant_demand_upd=pd.read_csv('plant_demand_upd.csv')
               # plant_demand_upd['grade_new']=plant_demand_upd['Grade']+'='+plant_demand_upd['Category']
                except:
                    plant_all_comb=pd.DataFrame()
#                    plant_demand_upd['Plant_key']=len(plant_demand['grade_new'].unique().tolist())*plant_demand['Plant_key'].unique().tolist()
#                    plant_demand_upd['grade_new']=len(plant_demand['Plant_key'].unique().tolist())*plant_demand['grade_new'].unique().tolist()
                    
                    b=plant_demand['Plant'].unique().tolist()
                    a1=plant_demand['grade_new'].unique().tolist()
                    plant_all_comb['combine']=[[x,y] for x in a1 for y in b]
                    plant_all_comb['grade_new']=plant_all_comb['combine'].str[0]
                    plant_all_comb['Plant']=plant_all_comb['combine'].str[1]
                    del plant_all_comb['combine']      
                    
                    plant_all_comb[['Grade', 'Category']]=plant_all_comb['grade_new'].str.split("=",expand=True)
                    plant_demand=plant_demand.groupby(['Plant', 'Grade', 'Category','grade_new'])['Demand_yearly'].sum().reset_index()
                    plant_demand_upd=pd.merge(plant_all_comb,plant_demand,how='left',on=['Plant', 'Grade', 'Category','grade_new'])
                    plant_demand_upd=plant_demand_upd.fillna(0)
                    
                    

                clear_linkagesp(52)
                st.markdown("<h1 style='text-align: center; color: red;'>Fly Ash Procurement Dashboard</h1>", unsafe_allow_html=True)
                #st.image(utc)
                st.markdown("<h2 style='text-align: center; color: black;'>Edit Existing Data</h2>", unsafe_allow_html=True)
                #st.sidebar.markdown(f"***Plant***")
                #creating dataframe for updating particular plant capacity
                df_category=pd.DataFrame()
                df_category['Category']=plant_demand['Category'].unique().tolist()+plant_demand['Category'].unique().tolist()
                df_category['Grade']=plant_demand['Grade'].unique().tolist()+plant_demand['Grade'].unique().tolist()+plant_demand['Grade'].unique().tolist()
                
                #select plant to change plant capacity
                st.subheader("**Plant Filters**")
                st.write("Select appropriate filters for plant")
                plant_demand_change=st.selectbox("Select a Plant",plant_demand['Plant'].unique().tolist())
                plant_demand_change_list=[plant_demand_change]
                #plant_demand_change_list=["APCW"]
                plant_demand1_change=plant_demand[plant_demand['Plant'].isin(plant_demand_change_list)]
                
                #make all combinations of selected plant
                df_category=pd.merge(df_category,plant_demand1_change[['Grade', 'Category', 'Demand_yearly']],how='left',on=['Category','Grade'])
                df_category=df_category.fillna(0)
#                st.write(df_category)
                #st.write(plant_demand1_change)
                
                
                #select material group (fly ash type)
                material_group_select=st.selectbox("Select a grade",df_category['Grade'].unique().tolist())
                material_group_select_list=[material_group_select]
                df_category=df_category[df_category['Grade'].isin(material_group_select_list)]
                #st.write("category percentage")
                #st.write("*Exisitng*")
                df_category['percentage']=(df_category['Demand_yearly']/df_category['Demand_yearly'].sum())*100
                plant_demand1_capacity=df_category['Demand_yearly'].sum()
                #st.write(df_category)
                #st.write("Existing Plant capacity "+str(round(plant_demand1_capacity,2)))
                
                #plant capacuty
                
                pc=st.text_input("Enter the Plant Demand",str(round(plant_demand1_capacity,2)))
                
                #best fly ash percentage
#                if best flyash isin df_categ
                best_fly_ash_perc=df_category[df_category['Category']=='Best Fly Ash']
                best_fly_ash_perc.index=range(0,len(best_fly_ash_perc))
                bp=st.text_input("Percentage of best fly ash ",str(best_fly_ash_perc['percentage'][0]))
                #avg fly ash percentage
                avg_fly_ash_perc=df_category[df_category['Category']=='Average Fly Ash']
                avg_fly_ash_perc.index=range(0,len(avg_fly_ash_perc))
                ap=st.text_input("Percentage of average fly ash ",str(avg_fly_ash_perc['percentage'][0]))
                #good fly ash percentage
                good_fly_ash_perc=df_category[df_category['Category']=='Good Fly Ash']
                good_fly_ash_perc.index=range(0,len(good_fly_ash_perc))
                gp=st.text_input("Percentage of good fly ash ",str(good_fly_ash_perc['percentage'][0]))
                
                #check percentage should be equal to 100 then proceed
                if float(bp)+float(ap)+float(gp)==100:
                    st.write("Plant Data:")
                    #make dataframe of all values
                    percentage_upd=[bp,ap,gp]
                    #percentage_upd=[80,20,0]
                    category_upd=['Best Fly Ash','Average Fly Ash','Good Fly Ash']
                    df_category_upd=pd.DataFrame()
                    df_category_upd['percentage']=percentage_upd
                    df_category_upd['Category']=category_upd
                    df_category_upd['Grade']=material_group_select
                    
                    df_category_upd['Demand_yearly']=float(pc)*(df_category_upd['percentage'].astype(float)/100)
                    df_category_upd['Plant']=plant_demand_change
                    df_category_upd_show=df_category_upd[['Category','Grade','Demand_yearly','percentage']]
                    df_category_upd_show=pd.merge(df_category_upd_show,df_category,how='left',on=['Category','Grade'])
                    df_category_upd_show.index=range(1,len(df_category_upd_show)+1)
                    df_category_upd_show=df_category_upd_show.rename(columns={'Demand_yearly_x':'Updated_Demand_yearly','Demand_yearly_y':'Existing_Demand_yearly','percentage_x':'Updated_percentage','percentage_y':'Existing_percentage'})
                    
                    st.write(df_category_upd_show)
                    #st.write("Existing Plant capacity "+str(round(plant_demand1_capacity,2)))
                    plant_demand_upd=pd.merge(plant_demand_upd,df_category_upd,how='left',on=['Plant', 'Grade', 'Category'])
                    plant_demand_upd['Demand_yearly_x']=np.where(plant_demand_upd['Demand_yearly_y'].notnull(),plant_demand_upd['Demand_yearly_y'],plant_demand_upd['Demand_yearly_x'])
                    plant_demand_upd=plant_demand_upd.rename(columns={'Demand_yearly_x':'Demand_yearly'})
                    del plant_demand_upd['Demand_yearly_y']
                    del plant_demand_upd['percentage']
                    if st.checkbox('Submit'):
                       # plant_demand_upd_show=plant_demand_upd[['category','material_group_new','Qty(LE)(LMT)','percentage']]
                        plant_demand_upd.to_csv(r'plant_demand_upd.csv',index=None)
                        #plant_demand_upd_show.index=range(1,len(plant_demand_upd_show)+1)
                        st.write("Updated")
                        #st.write(plant_demand_upd_show)
                else:
                    st.warning("Sum of percentage is not equal to 100 !")
                    
                #source capacity
                source_capacity[['Grade', 'Category']]=source_capacity['grade_new'].str.split("=",expand=True)
                try:
                    source_capacity_upd=pd.read_csv('source_capacity_upd.csv')
                except:
                    
                    source_all_comb=pd.DataFrame()
                    bs=source_capacity['Source'].unique().tolist()
                    as1=source_capacity['grade_new'].unique().tolist()
                    source_all_comb['combine']=[[x,y] for x in as1 for y in bs]
                    source_all_comb['grade_new']=source_all_comb['combine'].str[0]
                    source_all_comb['Source']=source_all_comb['combine'].str[1]
                    del source_all_comb['combine'] 
                    source_all_comb[['Grade', 'Category']]=source_all_comb['grade_new'].str.split("=",expand=True)
                    source_capacity=source_capacity.groupby(['Source_key','Source', 'Grade', 'Category','grade_new'])['Capacity_yearly'].sum().reset_index()
                    source_capacity_upd=pd.merge(source_all_comb,source_capacity,how='left',on=['Source', 'Grade', 'Category','grade_new'])
                    source_capacity_upd=source_capacity_upd.fillna(0)
                    
#                source_capacity[['Grade', 'Category']]=source_capacity['grade_new'].str.split("=",expand=True)
                
                st.subheader(f'***Source Filters***')
                #creating dataframe for updating particular plant capacity
                df_category_s=pd.DataFrame()
#                st.write(source_capacity)
                df_category_s['Category']=source_capacity['Category'].unique().tolist()+source_capacity['Category'].unique().tolist()
                df_category_s['Grade']=source_capacity['Grade'].unique().tolist()+source_capacity['Grade'].unique().tolist()+source_capacity['Grade'].unique().tolist()
                
                #select plant to change plant capacity
                source_capacity_change=st.selectbox("Select source",source_capacity['Source'].unique().tolist())
                source_capacity_change_list=[source_capacity_change]
                source_capacity1_change=source_capacity[source_capacity['Source'].isin(source_capacity_change_list)]
                
                #make all combinations of selected plant
                df_category_s=pd.merge(df_category_s,source_capacity1_change[['Grade', 'Category', 'Capacity_yearly']],how='left',on=['Category','Grade'])
                df_category_s=df_category_s.fillna(0)
                #st.write(df_category_s)
                
                #select material group (fly ash type)
                material_group_select_s=st.selectbox("Select material group for source",df_category_s['Grade'].unique().tolist())
                material_group_select_s_list=[material_group_select_s]
                df_category_s=df_category_s[df_category_s['Grade'].isin(material_group_select_s_list)]
                #st.write("category percentage")
                df_category_s['percentage']=(df_category_s['Capacity_yearly']/df_category_s['Capacity_yearly'].sum())*100
                source_capacity1=df_category_s['Capacity_yearly'].sum()
                #st.write(df_category_s)
                #st.write("Source capacity "+str(round(source_capacity1,2)))
                
                #plant capacity
                sc=st.text_input("Enter Source Capacity",str(round(source_capacity1,2)))
                
                df_category_s['percentage']=df_category_s['percentage'].fillna(0)
                #best fly ash percentage
                best_fly_ash_perc_s=df_category_s[df_category_s['Category']=='Best Fly Ash']
                best_fly_ash_perc_s.index=range(0,len(best_fly_ash_perc_s))
                bp_s=st.text_input("Percentage of best fly ash ",str(round(best_fly_ash_perc_s['percentage'][0])))
                #avg fly ash percentage
                avg_fly_ash_perc_s=df_category_s[df_category_s['Category']=='Average Fly Ash']
                avg_fly_ash_perc_s.index=range(0,len(avg_fly_ash_perc_s))
                ap_s=st.text_input("Percentage of average fly ash ",str(round(avg_fly_ash_perc_s['percentage'][0])))
                #good fly ash percentage
                good_fly_ash_perc_s=df_category_s[df_category_s['Category']=='Good Fly Ash']
                good_fly_ash_perc_s.index=range(0,len(good_fly_ash_perc_s))
                gp_s=st.text_input("Percentage of good fly ash ",str(round(good_fly_ash_perc_s['percentage'][0])),key=1)
                
                #check percentage should be equal to 100 then proceed
                if int(float(bp_s)+float(ap_s)+float(gp_s))==100:
                    
                    #make dataframe of all values
                    percentage_upd_s=[bp_s,ap_s,gp_s]
                    category_upd_s=['Best Fly Ash','Average Fly Ash','Good Fly Ash']
                    df_category_upd_s=pd.DataFrame()
                    df_category_upd_s['percentage']=percentage_upd_s
                    df_category_upd_s['Category']=category_upd_s
                    df_category_upd_s['Grade']=material_group_select_s
                    
                    df_category_upd_s['Capacity_yearly']=float(sc)*(df_category_upd_s['percentage'].astype(float)/100)
                    df_category_upd_s['Source']=source_capacity_change
                    #st.write(df_category_upd_s)
                    df_category_upd_s_show=df_category_upd_s[['Category','Grade','Capacity_yearly','percentage']]
                    df_category_upd_s_show=pd.merge(df_category_upd_s_show,df_category_s,how='left',on=['Category','Grade'])
                    df_category_upd_s_show.index=range(1,len(df_category_upd_s_show)+1)
                    df_category_upd_s_show=df_category_upd_s_show.rename(columns={'Capacity_yearly_x':'Updated_Capacity_yearly','Capacity_yearly_y':'Existing_Capacity_yearly','percentage_x':'Updated_percentage','percentage_y':'Existing_percentage'})
                    st.write("Source Data:")
                    st.write(df_category_upd_s_show)
    
                    
                    source_capacity_upd=pd.merge(source_capacity_upd,df_category_upd_s,how='left',on=['Source', 'Grade', 'Category'])
                    #st.write(source_capacity_upd)
                    source_capacity_upd['Capacity_yearly_x']=np.where(source_capacity_upd['Capacity_yearly_y'].notnull(),source_capacity_upd['Capacity_yearly_y'],source_capacity_upd['Capacity_yearly_x'])
                    source_capacity_upd=source_capacity_upd.rename(columns={'Capacity_yearly_x':'Capacity_yearly'})
                    st.write(source_capacity_upd)
                    del source_capacity_upd['Capacity_yearly_y']
                    del source_capacity_upd['percentage']
                    if st.checkbox('Submit',key=1):
                        source_capacity_upd.to_csv(r'source_capacity_upd.csv',index=None)
                        st.write("Updated")
                        #st.write(source_capacity_upd)
                else:
                    st.warning("Sum of percentage is not equal to 100 !")
                st.write("To update changes press submit")
                #Optimiser
                st.text("Please click on 'Run the Optimiser' present on the sidebar to your left to proceed")
                if st.sidebar.checkbox("Run the Optimizer"):
                    clear_linkagesp(79)
                    st.markdown("<h1 style='text-align: center; color: red;'>Fly Ash Procurement Dashboard</h1>", unsafe_allow_html=True)
                    st.markdown("<h2 style='text-align: center; color: black;'>Optimizer </h2>", unsafe_allow_html=True)
                    st.markdown('---')
                    try:
                        source_capacity=fileUpload_csv(r'source_capacity_upd.csv')
                    except:
                        source_capacity=source_capacity.copy()
                    try:
                        plant_demand=fileUpload_csv('plant_demand_upd.csv')
                    except:
                        plant_demand=plant_demand.copy()
                    #'''
                    #source code and plant code dataframe
                    #'''
                    source_code=df_input2[['source_name_given', 'Source_code']].copy()
                    source_code=source_code.drop_duplicates()
                    plant_code=df_input2[['Cluster/Unit', 'Cluster_code']]
                    plant_code=plant_code.rename(columns={'Cluster/Unit':'source_name_given','Cluster_code':'Source_code'})
                    plant_code=plant_code.drop_duplicates()
                    source_code=source_code.append(plant_code)
#                    
                    #ptpkm
                    ptpkm=pd.merge(ptpkm,source_code,how='left',on=['source_name_given'])
                    #handling
                    handling_cost=pd.merge(handling_cost, source_code, how='left',on=['source_name_given'])
                    #basic cost
                    basic_cost_ex=pd.merge(basic_cost_ex, source_code, how='left',on=['source_name_given'])
#                    #plant demand
#                    plant_demand_ex=pd.merge(plant_demand_ex,source_code,how='left',right_on=['source_name_given'],left_on=['Cluster/Unit'])
#                    #source capacity
#                    source_capacity_ex=pd.merge(source_capacity_ex,source_code,how='left',right_on=['source_name_given'],left_on=['source_name_given'])
#                    
                    df_dis['FromID']=df_dis['FromID'].astype(str)
                    df_dis['ToID']=df_dis['ToID'].astype(str)
                    file_lat_long_new=df_dis.copy()
                    df_dis_sp=df_dis[(df_dis['FromID'].str.contains('S')==True) & (df_dis['ToID'].str.contains('P')==True)]
                    
                    df_dis_sp['distance_slab']=pd.cut(df_dis_sp['Kilometres'],bins=[-1,100,200,300,400,500,600,700,9999],labels=['0-100','101-200','201-300','301-400','401-500','501-600','601-700','GT700'])
                    df_ptpkm_dis=pd.merge(df_dis_sp,ptpkm,how='left',left_on=['FromID','distance_slab'],right_on=['Source_code','distance_slab'])
            
#                    st.write("old",file_lat_long)
#                    st.write("new",file_lat_long_new)
                    #prep existinf function
                    data1=data1_prep(data,df_input,df_ptpkm_dis)

                    
#                    data2=data1[['source_name_given','Source_code','Cluster/Unit','material_group_new','category','grade_new','Qty(LE)(LMT)','Kilometres','ptpkm','freight_new','Basic Rate(Rs/MT)','Other(Rs/MT)','Total_cost']]
                    data1=data1[['source_name_given','Cluster/Unit','material_group_new','category','grade_new','Qty(LE)(LMT)','Kilometres','ptpkm','freight_new','Basic Rate(Rs/MT)','Other(Rs/MT)','Total_cost']]
                    data1=data1.reset_index(drop=True)
      
                    #plant demand, source capacity, handling cost and basic cost dictionary
                    plant_demand=plant_demand.dropna()
                    source_capacity=source_capacity.dropna()
                  

                    
                    grade=plant_demand['grade_new'].unique().tolist()
                    df4=pd.DataFrame()
                    df_final=pd.DataFrame()
                    plant_demand=pd.merge(plant_demand,key_plant,on="Plant",how='left')

                    df_final['OD']=[[k,j] for k in plant_demand['Plant_key'].unique().tolist() for j in source_capacity['Source_key'].unique().tolist()]
                    df_final['FromID']=df_final.apply(lambda OD: OD.str[0])
                    df4['FromID']=df_final['FromID'].copy()
                    df_final['FromID']=df_final.apply(lambda OD: OD.str[1])
                    df4['ToID']=df_final['FromID'].copy()
                    df4=pd.merge(df4,freight_cost[['Source_key','Plant_key','Final_logistics_cost(per ton)']],how='left',left_on=['ToID','FromID'],right_on=['Source_key','Plant_key'])
                    df4=df4[['ToID','FromID','Final_logistics_cost(per ton)']].copy()
                    
                    #df4["ToID"].replace(" ","-",regex=True,inplace=True)
                    df4=df4.dropna()
                    freight_dict=retro_dictify(df4[['ToID','FromID','Final_logistics_cost(per ton)']].copy())
                    d_prim=pd.DataFrame() 
                    st.markdown("*Select appropriate filters to view the optimized results*")
                    
#                    i="Flyash Dry=Average Fly Ash"
                    
                    for i in grade:
                        g=i
                        basic_cost_gr=basic_cost[basic_cost.grade_new==i].sort_values(['Source_key'])
                        basic_cost_gr.dropna(inplace=True)
                        #basic_cost_gr.drop_duplicates(['Source_key'],inplace=True)
                        basic_cost_gr_dict=retro_dictify(basic_cost_gr[["Source_key","Basic_cost"]])
                        plant_demand_gr=plant_demand[plant_demand.grade_new==i]
                        if plant_demand_gr.Demand_yearly.sum()==0:
                            continue
                        #freight_gr=freight_cost[freight_cost.grade_new==i]
                        source_capacity_gr=source_capacity[source_capacity.grade_new==i]
                        source_capacity_gr=source_capacity_gr[['Source_key','Capacity_yearly']].copy()
                        source_capacity_gr=source_capacity_gr.fillna(0).drop_duplicates()
                    #        source_capacity_gr=source_capacity_gr[['Sources','contract_Capacity (yearly)']]
                      
                        
                        #source_capacity_gr["Source"].replace(" ","-",regex=True,inplace=True)
                        source_capacity_gr_dict=retro_dictify(source_capacity_gr[['Source_key','Capacity_yearly']].copy())
                        plants=plant_demand_gr['Plant_key'].unique().tolist()
                        vendors=source_capacity_gr['Source_key'].unique().tolist()
                        plant_demand_gr=plant_demand_gr[['Plant_key','Demand_yearly']].copy()
                        plant_demand_gr.dropna(inplace=True)
                        plant_demand_gr_dict=retro_dictify(plant_demand_gr[['Plant_key','Demand_yearly']].copy())
                        routes_1=[(k,j) for k in vendors for j in plants]
                        #opti
                        prob=LpProblem("Transportation",LpMinimize,)
                        xi = LpVariable.dicts('xi',(vendors,plants) ,cat = 'Continuous',lowBound=0)
                     
                        prob += lpSum(xi[k][j]*(freight_dict[k][j]) for (k,j) in routes_1)+lpSum(xi[k][j]*basic_cost_gr_dict[k]  for (k,j) in routes_1)
                        #prob += lpSum(xi[i][j]*((freight_dict[i][j]) +basic_cost_gr_dict[i])  for (i,j) in routes_1)
                    
                        #prob += lpSum(xi[i][j]*(((ptpkm_ptpkm_dict[i][j]*ptpkm_km_dict[i][j])+handling_cost_dict[g][i])+basic_cost_dict[g][i]) for (i,j) in routes_1)
                        for k in vendors:
                            prob += lpSum(xi[k][j] for j in plants ) <= source_capacity_gr_dict[k]
                        
                        for j in plants:
                            prob += lpSum(xi[k][j] for k in vendors)==plant_demand_gr_dict[j]
                            
                        prob.solve()
                        p=[]
                        v=[]
                        q=[]    
                        for k in prob.variables():
                            if k.varValue!=0:
                                #print(i.name,"=", i.varValue)
                                t=k.name
                                s=t.split('_')
                                p.append(s[2])
                                v.append(s[1])
                                q.append(k.varValue)
                        d=pd.DataFrame(p)
                        d.rename(columns={0:"Plant_key"},inplace=True)
                        d['Source_key']=v
                        d['Quantity']=q
                        d['grade_new']=g
#                        d=pd.merge(d,key[["Plant_key","Plant"]],how='left',on=['Plant_key'])
#                        d=pd.merge(d,source_capacity[["Source_key","Source"]],how='left',on=['Source_key'])
#                        
                        d_prim=d_prim.append(d)
                    d_prim=d_prim.drop_duplicates().reset_index(drop=True)
                    d_prim=pd.merge(d_prim,freight_cost[["Source_key","Plant_key","final_rail_freight (per ton)","final_road_freight","Final_logistics_cost(per ton)","Mode_of_Transport"]],how='left',on=["Plant_key","Source_key"])
                    d_prim=pd.merge(d_prim,basic_cost[["Source_key","grade_new","Basic_cost"]],how='left',on=['Source_key','grade_new'])
#                    d_prim["Basic cost"].fillna(0,inplace=True)
                    d_prim["Total_cost"]=(d_prim["Basic_cost"]+d_prim["Final_logistics_cost(per ton)"])*d_prim["Quantity"]
                    d_prim['Total_cost (In Cr.)']=d_prim['Total_cost']/100
                    data1['Total_cost (In Cr.)']=data1['Total_cost']/100
                    d_summary=pd.DataFrame()
                    d_summary['Existing_cost (In Cr.)']=[round(data1['Total_cost'].sum()/100,0)]
                    d_summary['Reallocation_cost (In Cr.)']=[round(d_prim['Total_cost'].sum()/100,0)]
                    d_summary['Difference (In Cr.)']=[round((data1['Total_cost'].sum()-d_prim['Total_cost'].sum())/100,0)]
                    del d_prim['Total_cost']
                    del data1['Total_cost']  
#                    st.write(data1)
                    #map view
                    data_map_o=d_prim.copy()
                    data_map_o=pd.merge(data_map_o,lat_long,how='left',left_on=['Source_key'],right_on=['FromID'])
                    data_map_o=data_map_o.rename(columns={'FromLatitude':'Source_latitude','FromLongitude':'Source_longitude'})
                    del data_map_o['FromID']
                    data_map_o=pd.merge(data_map_o,lat_long,how='left',left_on=['Plant_key'],right_on=['FromID'])
                    data_map_o=data_map_o.rename(columns={'FromLatitude':'Plant_latitude','FromLongitude':'Plant_longitude'})
                    del data_map_o['FromID']
                    
                    data_map=df_input2.copy()
                    data_map=pd.merge(data_map,file_lat_long,how='left',left_on=['Source_code'],right_on=['FromID'])
                    data_map=data_map.rename(columns={'FromLatitude':'Source_latitude','FromLongitude':'Source_longitude'})
                    del data_map['FromID']
                    data_map=pd.merge(data_map,file_lat_long,how='left',left_on=['Cluster_code'],right_on=['FromID'])
                    data_map=data_map.rename(columns={'FromLatitude':'Cluster_latitude','FromLongitude':'Cluster_longiutde'})
                    del data_map['FromID']
                    
                    d_prim=pd.merge(d_prim,key_plant,how='left',on=["Plant_key"])
                    d_prim=pd.merge(d_prim,key_source,how='left',on=["Source_key"])
#                    d_prim=d_prim[["Source","Plant",]]
                    
                    
                    d_prim4=d_prim.copy()
                    d_prim4[['Grade','Category']]=d_prim4['grade_new'].str.split("=",expand=True)
                    data_map_o[['Grade','Category']]=data_map_o['grade_new'].str.split("=",expand=True)
                    data_map_o=pd.merge(data_map_o,key_plant,on=["Plant_key"],how='left')
                    data_map_o=pd.merge(data_map_o,key_source,on=["Source_key"],how='left')
                    
#                    st.write(data_map_o)
                    fly_ash_type=st.multiselect("Select Fly ash type",d_prim4['Grade'].unique().tolist(),key=2)
                    if len(fly_ash_type)!=0:
                        data_map1=data_map[data_map['material_group_new'].isin(fly_ash_type)]
                        data_map1=data_map1.reset_index(drop=True)
                        
                        data_map_o=data_map_o[data_map_o['Grade'].isin(fly_ash_type)]
                        data_map_o=data_map_o.reset_index(drop=True)
                    else:
                        data_map1=data_map.copy()
                        data_map1=data_map1.reset_index(drop=True)
                        
                        data_map_o=data_map_o.copy()
                        data_map_o=data_map_o.reset_index(drop=True)
                    source_list=st.multiselect("Select sources",d_prim4['Source'].unique().tolist(),key=2)
                    if len(source_list)!=0:
                        data_map1=data_map1[data_map1['source_name_given'].isin(source_list)]
                        data_map1=data_map1.reset_index(drop=True)
                        
                        data_map_o=data_map_o[data_map_o['Source'].isin(source_list)]
                        data_map_o=data_map_o.reset_index(drop=True)
                    else:
                        data_map1=data_map1.copy()
                        data_map1=data_map1.reset_index(drop=True)
                        
                        data_map_o=data_map_o.copy()
                        data_map_o=data_map_o.reset_index(drop=True)
                        
                    plant_list=st.multiselect("Select plant",d_prim4['Plant'].unique().tolist(),key=2)
                    if len(plant_list)!=0:
                        data_map1=data_map1[data_map1['Cluster/Unit'].isin(plant_list)]
                        data_map1=data_map1.reset_index(drop=True)
                        
                        data_map_o=data_map_o[data_map_o['Plant'].isin(plant_list)]
                        data_map_o=data_map_o.reset_index(drop=True)
                    else:
                        data_map1=data_map1.copy()
                        data_map1=data_map1.reset_index(drop=True)
                        
                        data_map_o=data_map_o.copy()
                        data_map_o=data_map_o.reset_index(drop=True)
                    category_list=st.multiselect("Select category",d_prim4['Category'].unique().tolist(),key=2)
                    if len(category_list)!=0:
                        data_map1=data_map1[data_map1['category'].isin(category_list)]
                        data_map1=data_map1.reset_index(drop=True)
                        
                        data_map_o=data_map_o[data_map_o['Category'].isin(category_list)]
                        data_map_o=data_map_o.reset_index(drop=True)
                    else:
                        data_map1=data_map1.copy()
                        data_map1=data_map1.reset_index(drop=True)
                        
                        data_map_o=data_map_o.copy()
                        data_map_o=data_map_o.reset_index(drop=True)
            
                    
#                    st.subheader("Data View")
                    
                    
                    d_prim4.index=range(1,len(d_prim4)+1)
                    d_prim4=d_prim4[["Source","Plant","Grade","Category","Quantity","final_rail_freight (per ton)","final_road_freight","Final_logistics_cost(per ton)","Mode_of_Transport","Basic_cost","Total_cost (In Cr.)","Source_key","Plant_key"]]
#                    st.write(d_prim4)
                    d_prim.index=range(0,len(d_prim))
                    data_map2_o=data_map_o[['Source_key','Source_latitude','Source_longitude','Plant_key','Plant_latitude','Plant_longitude']]
                    data_map2_o=data_map2_o.drop_duplicates()
                    df_s=data_map2_o[['Source_key','Source_latitude','Source_longitude']].copy()
                    df_s=df_s.drop_duplicates()
                    df_p=data_map2_o[['Plant_key','Plant_latitude','Plant_longitude']].copy()
                    df_p=df_p.drop_duplicates()
                    
                    
                    
                    
                    
                    
                    
                    data_map2=data_map1[['source_name_given','Source_latitude','Source_longitude','Cluster/Unit','Cluster_latitude','Cluster_longiutde','Qty(LE)(LMT)']]
        
                    df_s_e=data_map2[['source_name_given','Source_latitude','Source_longitude']].copy()
                    df_p_e=data_map2[['Cluster/Unit','Cluster_latitude','Cluster_longiutde']].copy()
                    
                    
                    
                    
                    
                    
                
                
                
                #st.write(data_map2)
                
                    s1=7
                    s2=9
                    c='blue'
                    n1='Sources'
                    n2='Plants'
                    c1='red'
                    c2='green'
                    cc='brown'
                    nl1="Existing"
                    nl2="Optimal"
                    
                    st.subheader("Map View")
        
                    data_map2.index=range(0,len(data_map2))
                    data_map2_o.index=range(0,len(data_map2_o))
#                    st.write(data_map1)
#                    st.write(data_map2)
                    fig_l2=line_plot2(data_map2,data_map2_o,df_s,df_p,s1,s2,c,n1,n2,c1,c2,cc,nl1,nl2,df_s_e,df_p_e)
                    st.plotly_chart(fig_l2)
                    data1=data1.drop(columns=["grade_new"])
                    data1=data1.rename(columns={"category":"Category","material_group_new":"Grade","source_name_given":"Source","Cluster/Unit":"Plant","material_group_new":"Grade","category":"Category","Qty(LE)(LMT)":"Quantity",'freight_new':'Freight_cost','Basic Rate(Rs/MT)':'Basic_cost'})

                    st.subheader("Results :")
                    d_summary.index=range(1,len(d_summary)+1)
                    st.write(d_summary)
                    d_summary.index=range(0,len(d_summary))
                    val = to_excel_result(d_summary,d_prim4,data1)
                    b64 = base64.b64encode(val)
                    href1= f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download the Result </a>' # decode b'abc' => abc
                    st.markdown(href1, unsafe_allow_html=True)
                    
                    if st.sidebar.checkbox("Results Summary"):
                        clear_linkagesp(107)
#                        clear_linkagesp(12)
                        st.markdown("<h1 style='text-align: center; color: red;'>Fly Ash Procurement Dashboard</h1>", unsafe_allow_html=True)
                        st.markdown("<h2 style='text-align: center; color: black;'>Results Summary </h2>", unsafe_allow_html=True)
                        st.markdown('---')
                        st.markdown("*Select appropriate filters to view results*")
                        source_select=st.multiselect("Select Source",d_prim4['Source'].unique().tolist())
                        if len(source_select)!=0:
                            data1=data1[data1["Source"].isin(source_select)]
                            d_prim4=d_prim4[d_prim4['Source'].isin(source_select)]
                        else:
                            data1=data1.copy()
                            d_prim4=d_prim4.copy()
                        
                        plant_select=st.multiselect("Select Plant",d_prim4['Plant'].unique().tolist())
                        if len(plant_select)!=0:
                            data1=data1[data1["Plant"].isin(plant_select)]
                            d_prim4=d_prim4[d_prim4['Plant'].isin(plant_select)]
                        else:
                            data1=data1.copy()
                            d_prim4=d_prim4.copy()
                        
                        grade_select=st.multiselect("Select Grade",d_prim4['Grade'].unique().tolist())
                        if len(grade_select)!=0:
                            data1=data1[data1["Grade"].isin(grade_select)]
                            d_prim4=d_prim4[d_prim4['Grade'].isin(grade_select)] 
                        else:
                            data1=data1.copy()
                            d_prim4=d_prim4.copy()
                        cat_select=st.multiselect("Select Category",d_prim4['Category'].unique().tolist())
                        if len(grade_select)!=0:
                            data1=data1[data1["Category"].isin(cat_select)]
                            d_prim4=d_prim4[d_prim4['Category'].isin(cat_select)]
                        else:
                            data1=data1.copy()
                            d_prim4=d_prim4.copy()     
                            
                        exis=data1.groupby(["Source","Plant","Grade","Category"]).sum().reset_index()
                        st.subheader("Existing data view")
                        st.write(exis)
                        st.subheader("Reallocated data view:")
                        final=d_prim4.groupby(["Source","Plant","Grade","Category"]).sum().reset_index()
                        st.write(final)
                        st.markdown('---')
#                        st.text("Thank you for using the dashboard! :)")
#                        st.markdown("<h2 style='text-align: center; color: black;'>Thank you for using the Dashboard!</h2>", unsafe_allow_html=True)
                        
                        












#    """Main function. Run this to run the app"""
#    st.sidebar.title("Layout and Style Experiments")
#    st.sidebar.header("Settings")
#    st.markdown(
#        """
## Layout and Style Experiments
#
#The basic question is: Can we create a multi-column dashboard with plots, numbers and text using
#the [CSS Grid](https://gridbyexample.com/examples)?
#
#Can we do it with a nice api?
#Can have a dark theme?
#"""
#    )
#
#    select_block_container_style()
#    add_resources_section()
#
#    # My preliminary idea of an API for generating a grid
#    with Grid("1 1 1", color=COLOR, background_color=BACKGROUND_COLOR) as grid:
#        grid.cell(
#            class_="a",
#            grid_column_start=2,
#            grid_column_end=3,
#            grid_row_start=1,
#            grid_row_end=2,
#        ).markdown("# This is A Markdown Cell")
#        grid.cell("b", 2, 3, 2, 3).text("The cell to the left is a dataframe")
#        grid.cell("c", 3, 4, 2, 3).plotly_chart(get_plotly_fig())
#        grid.cell("d", 1, 2, 1, 3).dataframe(get_dataframe())
#        grid.cell("e", 3, 4, 1, 2).markdown(
#            "Try changing the **block container style** in the sidebar!"
#        )
#        grid.cell("f", 1, 3, 3, 4).text(
#            "The cell to the right is a matplotlib svg image"
#        )
#        grid.cell("g", 3, 4, 3, 4).pyplot(get_matplotlib_plt())
#
#    st.plotly_chart(get_plotly_subplots())

# =============================================================================
# @st.cache
# def load_image(img):
# im =Image.open(os.path.join(img))
# return im
# @st.cache
# def load_files_gdna():
# b=load_image(r"gdnaf2.PNG")
# return b
# =============================================================================



main()