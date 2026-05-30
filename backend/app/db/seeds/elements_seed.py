from sqlalchemy import select

from services.embedding_service import get_embedding_service
from db.session import AsyncSessionLocal
from models.element import Element


ELEMENTS = [
    # =========================
    # RESISTORS
    # =========================
    {"name": "Resistor 10Ω", "amount": 2, "url": None,
     "comment": "Very low resistance resistor used for current sensing or power line protection."},

    {"name": "Resistor 22Ω", "amount": 1,
     "url": "https://botland.com.pl/rezystory-oporniki/8279-rezystor-tht-cf-weglowy-1w-22-200szt-5904422305505.html",
     "comment": "Low-value resistor used in LED drivers and signal conditioning."},

    {"name": "Resistor 47Ω", "amount": 32,
     "url": "https://botland.com.pl/rezystory-przewlekane/20008-rezystor-justpi-tht-cf-weglowy-14w-47-30szt-5904422328962.html",
     "comment": "Common resistor used for LED protection and digital signal limiting."},

    {"name": "Resistor 100Ω", "amount": 3,
     "url": "https://botland.com.pl/rezystory-przewlekane/20151-rezystor-justpi-tht-cf-weglowy-14w-100-30szt-5904422329297.html",
     "comment": "Used for moderate current limiting, especially LEDs and small loads."},

    {"name": "Resistor 220Ω", "amount": 34,
     "url": "https://botland.com.pl/rezystory-przewlekane/20014-rezystor-justpi-tht-cf-weglowy-14w-220-30szt-5904422329020.html",
     "comment": "Standard LED resistor for safe current limiting."},

    {"name": "Resistor 330Ω", "amount": 34,
     "url": "https://botland.com.pl/rezystory-przewlekane/20013-rezystor-justpi-tht-cf-weglowy-14w-330-30szt-5904422329013.html",
     "comment": "Typical LED and logic signal resistor."},

    {"name": "Resistor 470Ω", "amount": 34,
     "url": "https://botland.com.pl/rezystory-przewlekane/20016-rezystor-justpi-tht-cf-weglowy-14w-470-30szt-5904422329044.html",
     "comment": "Used in LED circuits and pull-down configurations."},

    {"name": "Resistor 1kΩ", "amount": 40,
     "url": "https://botland.com.pl/rezystory-przewlekane/20149-rezystor-justpi-tht-cf-weglowy-14w-10k-30szt-5904422329273.html",
     "comment": "General-purpose resistor used in biasing and digital circuits."},

    {"name": "Resistor 2.2kΩ", "amount": 4,
     "url": "https://botland.com.pl/rezystory-przewlekane/20015-rezystor-justpi-tht-cf-weglowy-14w-22k-30szt-5904422329037.html",
     "comment": "Pull-up / pull-down resistor commonly used in logic circuits."},

    {"name": "Resistor 4.7kΩ", "amount": 4,
     "url": "https://botland.com.pl/rezystory-przewlekane/20152-rezystor-justpi-tht-cf-weglowy-14w-47k-30szt-5904422329303.html",
     "comment": "Common pull-up resistor for I2C and digital inputs."},

    {"name": "Resistor 10kΩ", "amount": 2,
     "url": "https://botland.com.pl/rezystory-przewlekane/20150-rezystor-justpi-tht-weglowy-14w-10k-30szt-5904422329280.html",
     "comment": "Standard pull-up/down resistor used in digital electronics."},

    {"name": "Resistor 22kΩ", "amount": 4,
     "url": None,
     "comment": "Medium-high resistance used in biasing and voltage dividers."},

    {"name": "Resistor 47kΩ", "amount": 4,
     "url": "https://botland.com.pl/rezystory-przewlekane/3968-rezystor-tht-cf-weglowy-14w-47k-30szt-5904422305390.html",
     "comment": "High-value resistor used in signal conditioning and pull-ups."},

    {"name": "Resistor 0.1MΩ (100kΩ)", "amount": 4,
     "url": None,
     "comment": "High resistance used in sensing and low-current circuits."},

    {"name": "Resistor 0.22MΩ (220kΩ)", "amount": 4,
     "url": None,
     "comment": "High-value resistor used in analog filtering and bias networks."},

    {"name": "Resistor 0.47MΩ (470kΩ)", "amount": 4,
     "url": None,
     "comment": "Very high resistance used in low-current sensing applications."},

    {"name": "Resistor 1MΩ", "amount": 4,
     "url": "https://botland.com.pl/rezystory-przewlekane/20017-rezystor-justpi-tht-weglowy-14w-1m-30szt-5904422329051.html",
     "comment": "Very high resistance used in analog inputs and leakage control."},

    {"name": "Resistor 2.2MΩ", "amount": 2,
     "url": None,
     "comment": "Ultra-high resistance used in sensitive analog measurements."},

    {"name": "Resistor 4.7MΩ", "amount": 2,
     "url": None,
     "comment": "Extremely high resistance for low-current sensing circuits."},

    {"name": "Resistor 10MΩ", "amount": 2,
     "url": None,
     "comment": "Maximum range resistor used in very high impedance measurements."},

    # =========================
    # DIODES & LEDS
    # =========================
    {"name": "Diode 1N4007", "amount": 10,
     "url": "https://botland.com.pl/diody-prostownicze/1803-dioda-prostownicza-1n4007-1a-1000v-10szt-5903351244435.html",
     "comment": "General-purpose rectifier diode for power protection and AC rectification."},

    {"name": "Diode 1N4148", "amount": 3,
     "url": "https://botland.com.pl/diody-prostownicze/4927-dioda-prostownicza-1n4148-100v-015a-10szt-5903351244442.html",
     "comment": "Fast switching signal diode used in logic and high-speed circuits."},

    {"name": "LED 5mm Blue", "amount": 14,
     "url": "https://botland.com.pl/diody-led/19991-dioda-led-5mm-niebieska-10szt-justpi-5904422328795.html",
     "comment": "Blue indicator LED for status and UI signaling."},

    {"name": "LED 5mm White", "amount": 4,
     "url": "https://botland.com.pl/diody-led/3705-dioda-led-5mm-biala-zimna-clear-5szt-5903351244022.html",
     "comment": "White LED used for illumination or indicators."},

    {"name": "LED 5mm Red", "amount": 5,
     "url": "https://botland.com.pl/diody-led/19995-dioda-led-5mm-czerwona-10szt-justpi-5904422328832.html",
     "comment": "Red indicator LED for warnings and status signals."},

    {"name": "LED 5mm Yellow", "amount": 6,
     "url": "https://botland.com.pl/diody-led/19993-dioda-led-5mm-zolta-10szt-justpi-5904422328818.html",
     "comment": "Yellow LED used for alerts and intermediate status."},

    {"name": "LED 5mm Green", "amount": 6,
     "url": "https://botland.com.pl/diody-led/19994-dioda-led-5mm-zielona-10szt-justpi-5904422328825.html",
     "comment": "Green LED used for OK/status indicators."},

    {"name": "RGB LED 5mm Common Anode", "amount": 1,
     "url": "https://botland.com.pl/diody-led-rgb/543-dioda-led-5mm-rgb-wsp-anoda-5-szt-5903351244176.html",
     "comment": "RGB LED capable of displaying multiple colors via PWM control."},

    # =========================
    # ICs
    # =========================
    {"name": "CD4017BE Johnson Counter", "amount": 1,
     "url": "https://sklep-elwron.pl/pl/p/Uklad-scalony-cyfrowy-CD4017BE-licznik-Jonsona-DIP16/1898",
     "comment": "Decade Johnson counter used for sequential LED or signal control."},

    {"name": "CD40106BE Schmitt Inverter", "amount": 1,
     "url": "https://botland.com.pl/uklady-logiczne/11765-uklad-logiczny-cd40106be-6x-inwerter-z-przerzutnikiem-schmitta-5szt-5904422317980.html",
     "comment": "Hex inverter with Schmitt trigger for signal conditioning."},

    {"name": "CD4093BE NAND Schmitt Trigger", "amount": 1,
     "url": "https://botland.com.pl/uklady-logiczne/17147-uklad-logiczny-cd4093be-4xnand-z-przerzutnikiem-schmitta-5szt-5904422327019.html",
     "comment": "Quad NAND gate with Schmitt trigger inputs for noise-resistant logic."},

    {"name": "LM358N Operational Amplifier", "amount": 1,
     "url": "https://abc-rc.pl/pl/products/wzmacniacz-operacyjny-lm358n-3-32v-obudowa-dip8-11919.html",
     "comment": "Dual operational amplifier for analog signal processing."},

    # =========================
    # MOTORS & ACTUATORS
    # =========================
    {"name": "Servo SG90 Micro", "amount": 3,
     "url": "https://botland.com.pl/serwa-typu-micro/13128-serwo-sg-90-micro-180-5904422350338.html",
     "comment": "Small servo motor for lightweight mechanical positioning."},

    {"name": "Servo PowerHD HD-6001HB", "amount": 1,
     "url": "https://botland.com.pl/serwa-typu-standard/2311-serwo-powerhd-hd-6001hb-standard-6939570200913.html",
     "comment": "Standard-size servo motor for higher torque applications."},

    # =========================
    # SENSORS
    # =========================
    {"name": "Photoresistor GL5616", "amount": 3,
     "url": "https://botland.com.pl/fotorezystory/1564-fotorezystor-5-10k-gl5616-10szt-5903351245739.html",
     "comment": "Light-dependent resistor for measuring ambient light."},

    {"name": "DHT22 Temperature & Humidity", "amount": 1,
     "url": "https://botland.com.pl/czujniki-multifunkcyjne/2637-czujnik-temperatury-i-wilgotnosci-dht22-am2302-modul-przewody-5904422372712.html",
     "comment": "Accurate digital sensor for temperature and humidity monitoring."},

    {"name": "DHT11 Temperature & Humidity", "amount": 1,
     "url": "https://botland.com.pl/czujniki-multifunkcyjne/9301-czujnik-temperatury-i-wilgotnosci-dht11-50c-5904422372668.html",
     "comment": "Low-cost temperature and humidity sensor with moderate accuracy."},

    {"name": "MQ-3 Alcohol Sensor", "amount": 1,
     "url": "https://botland.com.pl/czujniki-gazow/3736-czujnik-alkoholu-mq-3-polprzewodnikowy-modul-niebieski-5904422359447.html",
     "comment": "Gas sensor for alcohol vapor detection."},

    {"name": "MQ-2 Gas Sensor", "amount": 1,
     "url": "https://botland.com.pl/czujniki-gazow/3027-czujnik-dymu-i-latwopalnych-gazow-mq-2-polprzewodnikowy-modul-niebieski-5904422359270.html",
     "comment": "Smoke and combustible gas detection sensor."},

    # =========================
    # SWITCHES & BUTTONS
    # =========================
    {"name": "Tact Switch 12x12mm", "amount": 4,
     "url": "https://botland.com.pl/tact-switch/11131-tact-switch-12x12mm-z-nasadka-kwadrat-niebieski-5szt-5904422307554.html",
     "comment": "Large tactile push button for user input interfaces."},

    {"name": "Tact Switch 6x6mm", "amount": 11,
     "url": "https://botland.com.pl/tact-switch/3495-tact-switch-6x6mm-5mm-tht-2pin-5szt-5904422307639.html",
     "comment": "Standard small tactile switch for PCB input buttons."},

    {"name": "Push Button 60mm Red", "amount": 1,
     "url": "https://botland.com.pl/przyciski-arcade-i-big-push-button-do-teleturnieju-familiada/10939-push-button-6cm-czerwony-5904422376628.html",
     "comment": "Large arcade-style button for physical user interaction."},

    # =========================
    # MICROCONTROLLERS
    # =========================
    {"name": "Arduino UNO R3", "amount": 1,
     "url": None,
     "comment": "Microcontroller board for prototyping and embedded systems."},

    {"name": "Arduino UNO REV3", "amount": 1,
     "url": None,
     "comment": "Alternative Arduino board for general-purpose development."},

    {"name": "USBasp AVR Programmer", "amount": 1,
     "url": "https://botland.com.pl/programatory/10793-programator-avr-zgodny-usbasp-isp-tasma-idc-czarny-5904422339333.html",
     "comment": "ISP programmer for AVR microcontrollers."},

    # =========================
    # CAPACITORS
    # =========================
    {"name": "Electrolytic Capacitor 1000uF 16V", "amount": 2,
     "url": "https://sklep.avt.pl/pl/products/kondensator-elektrolityczny-1000uf-16v-168130.html",
     "comment": "Large capacitor used for power smoothing and energy storage."},

    {"name": "Electrolytic Capacitor 100uF 35V", "amount": 10,
     "url": "https://botland.com.pl/kondensatory-elektrolityczne-tht/898-kondensator-elektrolityczny-100uf-35v-6x12mm-105c-tht-10szt-5903351248235.html",
     "comment": "Power supply filtering and decoupling capacitor."},

    {"name": "Ceramic Capacitor 100nF", "amount": 23,
     "url": "https://botland.com.pl/kondensatory-ceramiczne-tht/210-kondensator-ceramiczny-100nf50v-tht-10szt-5903351248198.html",
     "comment": "Standard decoupling capacitor for noise suppression."},

    {"name": "Ceramic Capacitor 22pF", "amount": 10,
     "url": "https://botland.com.pl/kondensatory-ceramiczne-tht/448-kondensator-ceramiczny-22pf-50v-tht-10-szt-5904422355609.html",
     "comment": "Used in crystal oscillator circuits."},

    # =========================
    # TRANSISTORS
    # =========================
    {"name": "BC547B NPN Transistor", "amount": 13,
     "url": "https://botland.com.pl/tranzystory-npn/254-tranzystor-bipolarny-npn-bc547b-50v-01a-5szt-5904422308025.html",
     "comment": "General-purpose NPN transistor for switching and amplification."},

    {"name": "BC557 PNP Transistor", "amount": 4,
     "url": "https://www.reichelt.com/pl/pl/tranzystor-pnp-to-92-45-v-0-1-a-0-5-w-bc-557b-p35845.html",
     "comment": "PNP transistor used for switching and signal amplification."},

    {"name": "BC556 PNP Transistor", "amount": 5,
     "url": "https://botland.com.pl/tranzystory-pnp/19033-tranzystor-bipolarny-pnp-bc556b-65v01a-5szt-5904422308056.html",
     "comment": "Low-noise PNP transistor for analog circuits."},

    # =========================
    # DISPLAY MODULES
    # =========================
    {"name": "LCD 2x16 Character Display", "amount": 1,
     "url": "https://botland.com.pl/wyswietlacze-alfanumeryczne-i-graficzne/19738-wyswietlacz-lcd-2x16-znakow-zielony-justpi-5903351243063.html",
     "comment": "Text display module for showing alphanumeric information."},

    {"name": "7-Segment Display 1-digit", "amount": 2,
     "url": "https://botland.com.pl/wyswietlacze-segmentowe-i-matryce-led/6441-wyswietlacz-8-segmentowy-x1-14mm-zolty-wsp-anoda-5904422357641.html",
     "comment": "Single-digit numeric display using LED segments."},

    {"name": "Nokia 5110 LCD Display", "amount": 1,
     "url": "https://botland.com.pl/wyswietlacze-alfanumeryczne-i-graficzne/2650-wyswietlacz-lcd-graficzny-84x48px-nokia-5110-niebieski-5904422309299.html",
     "comment": "Graphical display module with low power consumption."}
]

async def seed_elements():
    async with AsyncSessionLocal() as session:

        # already seeded
        result = await session.execute(select(Element.id).limit(1))
        if result.scalar_one_or_none():
            return

        embedding_service = get_embedding_service()
        items = []

        for element_data in ELEMENTS:
            embedding = embedding_service.embed(
                element_data["name"] + " " + (element_data.get("comment") or "")
            )

            element = Element(
                name=element_data["name"],
                amount=element_data["amount"],
                url=element_data.get("url"),
                comment=element_data.get("comment"),
                embedding=embedding
            )

            items.append(element)

        session.add_all(items)
        await session.commit()