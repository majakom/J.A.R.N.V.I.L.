from sqlalchemy import select

from db.session import AsyncSessionLocal
from models.yolo_class import YoloClass
from services.embedding_service import get_embedding_service


YOLO_CLASSES = {
    "1-5-Volt-Battery": "1.5V battery used as a low-voltage DC power source in electronics.",
    "3-3-Volt-Battery": "3.3V battery used for powering microcontrollers and low-power devices.",
    "7-Segment-Display": "LED display used to represent numeric digits in embedded systems.",
    "9-Volt-Battery": "Standard 9V battery used in Arduino projects and portable electronics.",

    "Arduino-Mega": "High-pin-count Arduino board for complex embedded and robotics projects.",
    "Arduino-Nano": "Compact Arduino board designed for breadboard prototyping.",
    "Arduino-Uno": "Standard Arduino microcontroller board used for learning and prototyping.",

    "BJT-Transistor": "Bipolar junction transistor used for switching and amplification.",
    "Bluetooth-Module": "Wireless module for Bluetooth communication between devices.",
    "Breadboard": "Reusable prototyping board for building temporary circuits without soldering.",
    "Bridge-Rectifier": "Circuit that converts AC to DC using diode bridge configuration.",
    "Buck-Converter": "DC-DC converter that efficiently steps down voltage.",

    "Buzzer": "Sound output component used for alerts and notifications.",

    "Capacitor-10mf": "10µF capacitor used for filtering and energy storage.",
    "Capacitor-470mf": "470µF capacitor used for power smoothing and stabilization.",

    "DC-Motor": "Electric motor that converts DC power into rotational motion.",
    "Diode": "Semiconductor device that allows current flow in one direction only.",

    "ESP32": "WiFi and Bluetooth-enabled microcontroller for IoT applications.",
    "ESP32-CAM": "ESP32 board with integrated camera for computer vision projects.",

    "FT-232-USB-Serial-Module": "USB-to-serial converter used for microcontroller programming.",

    "Film-Capacitor": "Stable capacitor used in analog and audio circuits.",
    "Fuse": "Protective component that prevents overcurrent damage.",
    "Fuse-Base": "Holder used for mounting and replacing fuses safely.",

    "GSM-Module": "Module enabling cellular communication (SMS, calls, data).",
    "Gas-Sensor": "Sensor that detects gases like smoke, LPG, or methane.",

    "Heat-Sink": "Component used to dissipate heat from electronic devices.",

    "High-Voltage-Ceramic-Capacitor": "Capacitor designed for high-voltage applications.",
    "Humidity-Sensor": "Sensor used to measure air humidity levels.",

    "IC-Base-14-Pin": "Socket for 14-pin integrated circuits.",
    "IC-Base-28-Pin": "Socket for 28-pin integrated circuits.",
    "IC-Chip": "Generic integrated circuit used in electronic systems.",

    "IGBT": "Power transistor used in high-voltage switching applications.",

    "IR-Sensor": "Infrared sensor used for object detection and tracking.",

    "Inductor": "Component that stores energy in a magnetic field.",

    "Keypad": "Matrix input device for entering numbers or commands.",

    "LCD-Display": "Display module used for showing text and simple graphics.",
    "LDR-Sensor": "Light-dependent resistor used to measure light intensity.",
    "LED-Light": "Light-emitting diode used for indicators and lighting.",

    "Low-Voltage-Ceramic-Capacitor": "Capacitor used for low-voltage filtering.",
    "MLC-Capacitor": "Multilayer ceramic capacitor used in compact electronics.",

    "MOSFET": "Efficient electronic switch used in power control systems.",

    "Motion-Sensor": "Sensor that detects movement in an area.",
    "Motor-Driver": "Module used to control motors (speed and direction).",

    "NTC-Thermistor": "Temperature-dependent resistor used for sensing heat.",

    "OLED-Display": "High-contrast low-power display module.",

    "Pin-Header": "Connector pins used for PCB and breadboard wiring.",

    "Push-Switch": "Momentary switch activated by pressing.",
    "RFID-Scanner": "Device used to read RFID tags for identification.",

    "Raindrops-Module": "Sensor that detects rain or water droplets.",
    "Relay-Module": "Electrically controlled switch for high-power circuits.",

    "Resistor": "Passive component used to limit current flow.",
    "Rocker-Switch": "Mechanical switch with toggle action.",

    "Servo-Motor": "Motor with precise angular position control.",

    "Soil-Moisture-Sensor": "Sensor that measures water content in soil.",
    "Sonar-Sensor": "Ultrasonic sensor used for distance measurement.",

    "TCRT5000": "Infrared reflective sensor used for line tracking.",
    "Tact-Switch": "Small tactile push-button switch.",

    "Taper-Potentiometer": "Variable resistor with logarithmic response.",
    "Trimmer-Potentiometer": "Adjustable resistor used for calibration.",

    "Water-Sensor": "Sensor that detects presence of water.",

    "Zener-Diode": "Diode used for voltage regulation and protection."
}

async def seed_yolo_classes():
    async with AsyncSessionLocal() as session:

        # already seeded
        result = await session.execute(select(YoloClass.id).limit(1))
        if result.scalar_one_or_none():
            return

        embedding_service = get_embedding_service()
        items = []

        for name, description in YOLO_CLASSES.items():
            embedding_input = f"{name} {description}"
            embedding = embedding_service.embed(embedding_input)

            items.append(
                YoloClass(
                    name=name,
                    description=description,
                    embedding=embedding
                )
            )

        session.add_all(items)
        await session.commit()