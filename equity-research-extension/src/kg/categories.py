"""Item -> product category (design doc §6.3) for the Phase 1 Input / Product / Technology nodes.

Rules on the item's canonical name, first match wins; order goes from specific to general (e.g. "memory test
equipment" is test_equipment, not memory_chips). Items no rule covers stay "unmapped" and are listed in the
coverage report; they are never guessed. Pure function of the name: no LLM, no documents.
"""
from __future__ import annotations

import re

CATEGORIES = ["raw_materials", "wafers_substrates", "process_chemicals", "industrial_specialty_gases",
              "equipment_subsystems", "fab_equipment", "test_equipment", "eda_ip", "logic_chips", "memory_chips",
              "analog_power_chips", "foundry_services", "packaging_osat", "components_passive_connectors",
              "optical_components", "pcb_boards", "servers_systems", "networking_equipment", "storage", "devices",
              "compute_capacity", "cloud_services", "software_licenses", "ems_odm_services", "other_off_chain"]

LABEL = {"eda_ip": "EDA & IP", "packaging_osat": "packaging & test (OSAT)", "analog_power_chips": "analog & power chips",
         "ems_odm_services": "EMS / ODM services", "pcb_boards": "PCBs", "components_passive_connectors": "passive components",
         "industrial_specialty_gases": "specialty gases", "wafers_substrates": "wafers & substrates",
         "other_off_chain": "other products"}


def label(cat: str) -> str:
    return LABEL.get(cat, cat.replace("_", " "))


RULES: list[tuple[str, str]] = [  # (category, regex on the lower-cased name)
    # misfires found in the 16 Nov spot check (NVIDIA, TSMC, Micron, ASML, Broadcom, AMD profiles)
    ("logic_chips", r"\bgpus?\b"),                                   # "discrete GPUs" is not a discrete component
    ("packaging_osat", r"packaging (technolog|and testing)|chip assembly|chip testing services"),
    ("foundry_services", r"process technolog|process nodes?|lithography processes|semiconductor manufacturing services|"
                         r"multi-project wafer|foundry"),
    ("fab_equipment", r"steppers?"),
    ("eda_ip", r"design for manufacturing"),
    ("optical_components", r"lasers?"),
    ("networking_equipment", r"fibre channel"),
    ("analog_power_chips", r"acoustic resonator|radio[- ]frequency|\brf\b"),
    # specific names seen in the Phase 1 graph that the general rules below would miss or misfile
    ("test_equipment", r"yield management|process control|film thickness|measurement systems|hardware emulation|"
                       r"profiling systems|analysis systems"),
    ("process_chemicals", r"deposition materials"),
    ("packaging_osat", r"bonding systems|dispensing systems|dicing|lapping|ceramic packages"),
    ("fab_equipment", r"thin film process equipment|cleaning and surface preparation"),
    ("equipment_subsystems", r"spare parts"),
    ("wafers_substrates", r"pellicles?"),
    ("eda_ip", r"signoff|implementation tools|core libraries|tool libraries"),
    ("logic_chips", r"pcie controllers?"),
    ("test_equipment", r"interferometry"),
    ("raw_materials", r"crucibles?|polyvinyl"),
    ("packaging_osat", r"multi-chip packages?"),
    ("analog_power_chips", r"power (integrated circuits|electronics)|power and signal chain|acoustic wave filters|"
                           r"wi-fi and bluetooth"),
    ("memory_chips", r"\bmram\b|magnetoresistive|\bram\b"),
    ("logic_chips", r"integrated circuit dies"),
    ("raw_materials", r"polymers?|resins?|coatings?|graphite"),
    ("industrial_specialty_gases", r"\bgas\b"),
    ("equipment_subsystems", r"electrostatic chucks?|reticle pods?|materials delivery systems"),
    ("components_passive_connectors", r"attenuators?|^resonators?$"),
    ("cloud_services", r"data centers?\b|data center facilities"),
    ("other_off_chain", r"streaming|payment|app stores?|retail|weather|power information|military|weapons|guidance|"
                        r"submarine|landing systems|heat pumps?|smart home|telematics|fleet"),
    ("fab_equipment", r"thermal processing|wet clean|dry strip|bevel clean|electroplating|plasma etch|"
                      r"display manufacturing systems|radiation sources"),
    ("equipment_subsystems", r"equipment (spare parts|maintenance)"),
    ("packaging_osat", r"lead ?frames?|wirebond|system-in-package|package-in-package|package-on-package|fan-out|"
                       r"molding compounds?"),
    ("wafers_substrates", r"mask making"),
    ("raw_materials", r"\bgold\b|sputtering targets?|dielectric films?"),
    ("memory_chips", r"sdram|\bddr\b|gddr"),
    ("analog_power_chips", r"micro-electro-mechanical|radio-frequency|acoustic resonator|optocouplers?|"
                           r"touch controllers?|smartcard ics?|serializer|transistors?"),
    ("logic_chips", r"processing units?|\bics?\b|chipsets?|tensor processing|arithmetic logic|computing platforms?|"
                    r"accelerated computing|chiplets?|desktop platforms?"),
    ("networking_equipment", r"telecommunications equipment|base stations?|antennas?|wireless hardware|multiplexers?|"
                             r"connectivity (solutions|systems)|host bus adapters?|modem"),
    ("eda_ip", r"licen[cs]es?|design expertise|toolkits?|templates"),
    ("software_licenses", r"firmware|ai models?|foundation models?"),
    ("components_passive_connectors", r"batter(y|ies)"),
    ("devices", r"televisions?|flat panel displays?|computer monitors?|soundbars?|set-top boxes|media players?|gaming hardware|"
                r"calculators?|smartcards?$|appliances?|\b(desktop|tablet|personal) computers?\b|toys?|synthesizers?|"
                r"transistor radios|radio receivers|fax|satellite receivers?"),
    ("test_equipment", r"\btest(er|ers|ing)?\b.*\b(equipment|systems?|handlers?|sockets?)|\bprobers?\b|probe cards?|"
                       r"metrology|inspection|ellipsomet|\bate\b|burn-in|review systems?"),
    ("equipment_subsystems", r"subsystems?|components? for (semiconductor|fab)|equipment components|vacuum (pumps?|valves?)|"
                             r"rf power|power delivery (systems|for)|gas delivery|fluid delivery|mass flow|"
                             r"chambers? (parts|components)|optical columns|precision motion|robots?|handling"),
    ("fab_equipment", r"lithograph|scanners?\b|\betch(ing)?\b.*(equipment|systems?|tools?)|deposition|\bcvd\b|\bpvd\b|\bald\b|"
                      r"\bmbe\b|epitax|ion implant|implanters?|annealing|\bcmp\b (equipment|systems)|planarization (equipment|systems)|"
                      r"wafer (fab|processing) equipment|semiconductor (manufacturing|processing|fabrication) equipment|"
                      r"fab equipment|track equipment|crystal growth|mask-making|material removal|cleaning (equipment|systems)|"
                      r"\bequipment\b.*semiconductor|wafer fabrication equipment|ion beam"),
    ("industrial_specialty_gases", r"\bgases?\b|nitrogen|argon|helium|neon|hydrogen|fluorine"),
    ("process_chemicals", r"photoresists?|chemistr|chemicals?|slurr|cleaners?|precursors?|solvents?|"
                          r"(cmp|planarization) materials|etchants?|developers?|wet process|filtration|specialty materials"),
    ("wafers_substrates", r"wafers?\b(?!.*(equipment|inspection|fab))|substrates?|silicon carbide|\bsic\b|gallium nitride|"
                          r"\bgan\b(?!.*(chip|transistor|device))|sapphire|quartz|polysilicon|ingots?|photomasks?|masks? blanks?"),
    ("raw_materials", r"raw materials?|metals?\b|minerals?|rare earth|ceramic materials|copper|cobalt|tungsten|"
                      r"semiconductor materials|packaging materials|thermal interface materials|materials?$"),
    ("eda_ip", r"\beda\b|electronic design automation|design (software|tools|enablement|ecosystems?|services)|"
               r"\bip\b|intellectual property|\bip cores?|physical implementation|cores? licens|architecture licens|"
               r"instruction set|\barm\b|verification|simulation software|microarchitectures?"),
    ("foundry_services", r"foundry|process technolog|process nodes?|\bnm\b|nanometric|wafer manufacturing services|"
                         r"fdx|finfet|\bcmos\b.*process|fabrication services|manufacturing services for (chips|semiconductors)"),
    ("packaging_osat", r"packaging|assembly (and test|services)|\bosat\b|bump|flip.?chip|chiplets? integration|"
                       r"advanced packages?|quad flat|\bcowos\b|interposers?|test services"),
    ("memory_chips", r"\bdram\b|\bnand\b|\bhbm\b|flash|memory|\bsram\b|\bnor\b|eeprom"),
    ("analog_power_chips", r"analog|mixed-signal|power (management|semiconductors?|chips?|ics?|devices|transistors|modules|stages)|"
                           r"\bmosfets?\b|\bigbts?\b|motor drivers?|output power stages|\brf\b|radio frequency|amplifiers?|"
                           r"sensors?|timing|mems|data converters?|interface (ics?|integrated circuits)|gallium arsenide|"
                           r"switch(es)?\b(?!.*network)|discretes?|diodes?|voltage regulators?|input-output expander|"
                           r"antenna chips?|image sensors?|radar"),
    ("logic_chips", r"\bgpus?\b|\bcpus?\b|processors?|microcontrollers?|\bmcus?\b|\bsocs?\b|system[- ]on[- ]chip|"
                    r"\basics?\b|\bfpgas?\b|accelerators?|chips?\b|integrated circuits|semiconductors?\b|"
                    r"semiconductor products|digital micromirror|\bdsps?\b|graphics"),
    ("optical_components", r"optic|photonic|lasers?|transceivers?|fiber|\bleds?\b|lens"),
    ("components_passive_connectors", r"capacitors?|resistors?|inductors?|passive components|connectors?|cables?|"
                                      r"crystals?\b|oscillators?|components, modules"),
    ("pcb_boards", r"printed circuit|\bpcbs?\b|circuit boards?|motherboards?|boards?\b"),
    ("networking_equipment", r"network(ing)?|switch(ing)? (systems|platforms|silicon)|routers?|ethernet|interconnects?|"
                             r"infiniband|\bnics?\b|network interface"),
    ("storage", r"storage|solid[- ]state drives?|\bssds?\b|hard (disk )?drives?|\bhdds?\b"),
    ("servers_systems", r"servers?|data center (systems|infrastructure)|supercomputers?|workstations?|compute subsystems?|"
                        r"\bsystems\b.*\b(ai|computing)\b|rack"),
    ("compute_capacity", r"cloud instances?|compute capacity|gpu cloud|computing capacity"),
    ("cloud_services", r"cloud|hosting|data center services|colocation"),
    ("software_licenses", r"software|operating systems?|platforms? software|applications?\b|algorithms?|\bsdks?\b|drivers?\b"),
    ("ems_odm_services", r"contract manufactur|\bems\b|\bodm\b|electronics manufacturing services|manufacturing services"),
    ("devices", r"smartphones?|phones?|laptops?|\bpcs?\b|personal computers?|tablets?|consoles?|wearables?|printers?|"
                r"devices?|automotive electronics|consumer electronics|radio receivers|enclosures|vehicles?|cameras?"),
]
NEVER = re.compile(r"standards?$|\bbrands?\b|portfolio of products|other offerings|services and support|"
                   r"customer services|technical support")
_COMPILED = [(c, re.compile(p)) for c, p in RULES]


def categorize(name: str) -> str | None:
    s = (name or "").lower()
    if NEVER.search(s):
        return None
    for cat, pat in _COMPILED:
        if pat.search(s):
            return cat
    return None
