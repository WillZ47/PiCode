import time
import cv2 as cv  
import paho.mqtt.client as mqtt
import statistics 
import sqlite3  
from ultralytics import YOLO
import pandas as pd  
import os 
import pathlib



class ThresholdDataBase: 
    def __init__(self, a_db_name, a_table_name):  
        self.db_name=a_db_name 
        self.table_name=a_table_name 
        self.connection = sqlite3.connect(self.db_name)  
        self.cursor = self.connection.cursor()

        self.cursor.execute(f'CREATE TABLE IF NOT EXISTS {self.table_name} (id INTEGER AUTO_INCREMENT PRIMARY KEY, avg_person_count INTEGER)')
                
        self.close_db()
    
    def open_db(self): 
        self.connection = sqlite3.connect(self.db_name)  
        self.cursor = self.connection.cursor()  
    
    def close_db(self): 
        self.connection.commit() 
        self.cursor.close()
        self.connection.close()
       
    def write_threshold(self, avg_person_count): 
        self.open_db()  

        self.cursor.execute(f'INSERT INTO {self.table_name} (avg_person_count) VALUES ({avg_person_count})')

        self.close_db()

    def get_threshold(self, thresh_type):
        self.open_db() 
        
        if(thresh_type=='max'):self.cursor.execute(f'SELECT * FROM {self.table_name} ORDER BY avg_person_count DESC LIMIT 1')  
        elif (thresh_type=='min'): self.cursor.execute(f'SELECT * FROM {self.table_name} ORDER BY avg_person_count ASC LIMIT 1')  
        
        row=self.cursor.fetchone() 
        
        self.close_db ()

        ''' 
        - 0:ID 
        - 1:avg_person_ct
        '''

        return float(row[1])
    

class MQTTClient: 
    def __init__(self, a_checker, a_broker_url="", a_broker_port=0, a_username="", a_password="",a_publish_rate=1, a_state="ACTIVE", a_reciever_topic='test/topic'): 
        
        self.publish_rate=a_publish_rate
        self.client = mqtt.Client()
        self.client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)  # Use secure TLS/SSL
        self.client.tls_insecure_set(False)
        self.client.username_pw_set(a_username, a_password)
        self.client.connect(a_broker_url, a_broker_port)
        self.client.loop_start()  

        #On_Message related code.
        self.checker=a_checker
        self.client.on_message=self.check_message 
        self.client.subscribe(a_reciever_topic)
        self.state=a_state

    def publish(self,topic,message): 
        print(f"published {message} on {topic}")
        self.client.publish(topic, message)  

    def switch_state(self):  
        if self.state=="ACTIVE":self.state="INACTIVE"
        elif self.state=="INACTIVE":self.state="ACTIVE" 

    def check_message(self, client, userdata, msg): 
        if self.checker(msg.payload.decode()): self.switch_state() 
    
    def __del__(self): 
        self.client.loop_stop() 
        self.client.disconnect()


class NonBlockingTimer:
    def __init__(self): 
        self._current_time=time.time()
        
    def nonBlock(self, logic, time_interval, **kwargs): 
        ''' 
        Function: nonBlock(self, logic, time_interval, **kwargs)
        
            -nonblocking time, accessed via callback.

        Returns: [VOID]
        ''' 
        new_current_time=time.time()

        if new_current_time-self._current_time>time_interval:  
            logic(**kwargs) 
            self._current_time=new_current_time 


class AlgorithmObject: 
    def __init__(self, a_camera=0, a_capture_rate=2, a_model="", a_database=None, a_priority="speed", a_location='Geisel Library'): 
        self.cam=cv.VideoCapture(a_camera) 
        self.capture_rate=a_capture_rate  
        self.model=YOLO(a_model) 
        self.database=a_database 
        self.priority=a_priority #the collection type to prioritize
        self.person_ct=[] #the moving average
        self.max=0 #the max collected since write_thresh_to_db(self) was called
        self.person_risk_lvl=['VERY LOW','LOW','MEDIUM','HIGH','VERY HIGH'] 
        self.location=a_location
       
    def capture_people(self): 
        '''  
        Function: capture_people(self) 
        
            -Captures the current amount of people and puts them on a moving average. 
            -Window size = 5 
            -This is done by accessing the boxes and cls attribute of a results object, and then having the cls id mapped to a string in a pandas dataframe.  
            -can be prioritized for speed and accuracy

        Returns: [VOID]
        '''
        ret, frame =self.cam.read()   
        if(self.priority=="speed"):  results = self.model(frame) 
        elif(self.priority=="accuracy"): results = self.model.track(frame, persist=True) 

        temp_ct=0 

        try: temp_ct=pd.Series(list(map(int, results[0].boxes.cls.tolist()))).value_counts().rename(index=results[0].names)['person'] 
        except(KeyError): temp_ct=0
        
        self.person_ct.append(int(temp_ct))  
        
        print(f'current person_ct: {temp_ct}')  
        print(f'current person_ct(list):{self.person_ct}')
        print(f'current person_avg: {statistics.mean(self.person_ct)}')

        if(len(self.person_ct)==5):   
            moving_avg=statistics.mean(self.person_ct) 
            if(moving_avg>self.max):self.max=moving_avg 

            self.person_ct.clear()
            self.person_ct.append(moving_avg) 
            
    
    def write_thresh_to_db(self):  
        '''  
        Function: write_thresh_to_db(self) 
        
            -Writes the maximum average value that the object has collected since the last time this function was called to the database.
        
        Returns: [VOID]
        '''
        if(self.database is not None):
            self.database.write_threshold(avg_person_count=self.max) 
            print(f'successfully wrote {statistics.mean(self.person_ct)} to {self.database.db_name} in table {self.database.table_name}') 
            self.max=0
            self.person_ct.clear()
            

    def calculate_risk(self):  
        '''  
        Function: calculate_risk(self)  
        
            -gets the current average and divides it by the difference of the higest and lowest averages from the database.
            -number is rounded and capped off at the length of the self.person_risk_lvl_attribute 
        
        Returns: The risk level(see the self.person_risk_lvl attribute), and the numeric risk level
        '''
        max_person_ct=self.database.get_threshold('max') 
        min_person_ct=self.database.get_threshold('min') 
        
        risk_lvl=(statistics.mean(self.person_ct)/(max_person_ct-min_person_ct))*len(self.person_risk_lvl)
        if(round(risk_lvl)>(len(self.person_risk_lvl)-1)): risk_lvl=len(self.person_risk_lvl)-1


        return f' {{\"risk_lvl_text\": \"{self.person_risk_lvl[round(risk_lvl)]}\", \"risk_lvl\": {risk_lvl}, \"location\": \"{self.location}\"}}'

        
LOCATION_TAG='Geisel Library' 
MAIN_TOPIC='test/topic'

myNBT=NonBlockingTimer()  
myNBT2=NonBlockingTimer() 
myNBT4=NonBlockingTimer() 
myThresholdDB=ThresholdDataBase ( 
                                a_db_name=f"{pathlib.Path(__file__).resolve().parent}/threshold.db",
                                a_table_name="thresholds"
                                )

myAO=AlgorithmObject            (
                                a_camera=0,
                                a_capture_rate=5, 
                                a_model=f'{pathlib.Path(__file__).resolve().parent}/yolov8n.pt', 
                                a_priority="speed",
                                a_database=myThresholdDB, 
                                a_location=LOCATION_TAG
                                )

myMQ=MQTTClient                 (
                                a_broker_url="a98bdda5eadc4d9db9ad2f32aceb4ae4.s1.eu.hivemq.cloud",  
                                a_broker_port=8883, 
                                a_username= "hivemq.webclient.1714355997131", 
                                a_password="D:k.0tg9@a53bJCB!uMO",  
                                a_publish_rate=5,
                                a_checker = lambda x: x == LOCATION_TAG, 
                                a_reciever_topic=MAIN_TOPIC
                                )

while True:  
    myAO.capture_people() 

    if myMQ.state == 'ACTIVE':
        myNBT2.nonBlock(logic=myMQ.publish, time_interval=myMQ.publish_rate,  topic=MAIN_TOPIC, message=myAO.calculate_risk()) 
    elif myMQ.state =='INACTIVE': 
        myNBT.nonBlock(logic = lambda :print("INACTIVE"), time_interval=2)

    myNBT4.nonBlock(logic=myAO.write_thresh_to_db, time_interval=300)
    
    
    


    