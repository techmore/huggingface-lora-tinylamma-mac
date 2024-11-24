import random
import json

meraki_templates = {
    "devices": ["MR46E", "MS120", "MX250", "MC74"],
    "features": ["VLANs", "QoS", "ACLs", "Firewall", "VPN", "RADIUS"],
    "locations": ["Library", "Classroom", "Lab", "Office", "Cafeteria"],
    "issues": ["configuration", "performance", "security", "connectivity"],
    "purposes": ["student access", "staff network", "guest network", "IoT devices"],
    "security": ["802.1x", "WPA3", "MAC filtering", "IPS", "Content filtering"],
    "instructions": [
        "Configure {feature} on {device} in {location} for {purpose}",
        "Set up {security} on {device} for {purpose}",
        "Optimize {device} {feature} settings for {purpose}",
        "Troubleshoot {issue} with {device} {feature}"
    ]
}

google_workspace_templates = {
    "services": ["Google Classroom", "Google Drive", "Admin Console", "Chrome Management"],
    "features": ["sharing settings", "access control", "user management", "device policies"],
    "usergroups": ["students", "teachers", "staff", "administrators"],
    "security": ["2FA", "security keys", "access restrictions", "audit logs"],
    "tasks": ["configuration", "deployment", "management", "monitoring"],
    "purposes": ["remote learning", "collaboration", "content filtering", "device management"],
    "instructions": [
        "Set up {service} {feature} for {usergroup}",
        "Configure {security} in {service} for {usergroup}",
        "Deploy {service} {feature} for {purpose}",
        "Manage {service} {aspect} for {usergroup}"
    ]
}

macos_mdm_templates = {
    "software": ["Jamf Pro", "Mosyle", "Apple School Manager", "Profile Manager"],
    "devices": ["MacBook Air", "MacBook Pro", "iMac", "Mac mini"],
    "settings": ["profiles", "restrictions", "configurations", "policies"],
    "features": ["enrollment", "deployment", "updates", "security"],
    "security": ["FileVault", "Gatekeeper", "XProtect", "Firewall"],
    "instructions": [
        "Deploy {software} {setting} to {device}",
        "Configure {feature} for {device} using {software}",
        "Set up {security} on {device} through {software}",
        "Manage {setting} for {device} deployment"
    ]
}

chrome_templates = {
    "features": ["enrollment", "policies", "extensions", "updates"],
    "settings": ["security", "content filters", "user permissions", "network"],
    "aspects": ["management", "deployment", "monitoring", "restrictions"],
    "security": ["Safe Browsing", "password manager", "site isolation", "sandboxing"],
    "instructions": [
        "Configure Chrome {feature} for {usergroup}",
        "Set up {setting} policies for {purpose}",
        "Deploy Chrome {aspect} in {location}",
        "Manage Chrome {feature} for {software}"
    ]
}

# Base templates for different categories
meraki_templates_original = {
    "instructions": [
        "Configure Meraki {device} for {location}",
        "Troubleshoot {device} {issue} in {location}",
        "Optimize {device} settings for {purpose}",
        "Set up {feature} on Meraki {device}",
        "Implement {security} for Meraki {device}"
    ],
    "devices": ["MR46E", "MS120", "MX250", "MR44", "MR36"],
    "locations": ["classroom", "library", "gym", "cafeteria", "admin office", "STEM lab", "media center"],
    "issues": ["connectivity", "performance", "interference", "configuration", "security"],
    "features": ["VLANs", "QoS", "content filtering", "guest access", "traffic shaping"],
    "purposes": ["high-density usage", "video streaming", "testing environment", "BYOD support", "secure access"],
    "security": ["802.1X", "NAC", "IPS", "content filtering", "access policies"]
}

google_workspace_templates_original = {
    "instructions": [
        "Configure {service} for {purpose}",
        "Set up {feature} in {service}",
        "Manage {service} {aspect} for {usergroup}",
        "Implement {security} for {service}",
        "Automate {task} in {service}"
    ],
    "services": ["Google Classroom", "Google Drive", "Gmail", "Google Calendar", "Google Meet", "Admin Console"],
    "features": ["sharing settings", "access controls", "retention policies", "sync settings", "integration"],
    "purposes": ["remote learning", "staff collaboration", "student projects", "parent communication", "administrative tasks"],
    "usergroups": ["teachers", "students", "staff", "administrators", "department heads"],
    "security": ["2FA", "data loss prevention", "audit logging", "access controls", "compliance settings"],
    "tasks": ["account provisioning", "resource management", "license assignment", "policy deployment", "backup procedures"]
}

macos_mdm_templates_original = {
    "instructions": [
        "Deploy {software} to {device} via MDM",
        "Configure {setting} on {device}",
        "Manage {feature} for {usergroup}",
        "Implement {security} on {device}",
        "Automate {task} for {device}"
    ],
    "software": ["Adobe Creative Cloud", "Microsoft Office", "security tools", "educational software", "productivity apps"],
    "devices": ["MacBook Air", "iMac", "Mac Pro", "Mac mini", "MacBook Pro"],
    "settings": ["FileVault", "network profiles", "printer configuration", "software updates", "login window"],
    "features": ["profiles", "configurations", "restrictions", "app management", "updates"],
    "security": ["disk encryption", "access controls", "security policies", "compliance checks", "password policies"]
}

chrome_templates_original = {
    "instructions": [
        "Configure {feature} for {purpose} on Chromebooks",
        "Set up {setting} for {usergroup}",
        "Manage {aspect} in {location}",
        "Implement {security} for Chrome devices",
        "Deploy {software} to {usergroup}"
    ],
    "features": ["kiosk mode", "device policies", "enrollment settings", "user policies", "app deployment"],
    "settings": ["update policies", "user settings", "device settings", "network configs", "security policies"],
    "aspects": ["device enrollment", "user profiles", "app management", "policy enforcement", "updates"],
    "security": ["content filtering", "extension policies", "URL blocking", "safe browsing", "device restrictions"]
}

def generate_instruction(template, **kwargs):
    return template.format(**kwargs)

def generate_input_context():
    contexts = [
        "Current setup needs update to support {number} users",
        "Experiencing issues with {aspect} during peak hours",
        "Need to ensure compliance with {standard}",
        "Planning for {event} next month",
        "Current solution is {issue} and needs improvement"
    ]
    numbers = ["50", "100", "200", "500", "1000"]
    aspects = ["performance", "reliability", "security", "usability", "compatibility"]
    standards = ["COPPA", "FERPA", "HIPAA", "PCI", "state requirements"]
    events = ["standardized testing", "new semester", "system upgrade", "audit", "staff training"]
    issues = ["outdated", "inefficient", "insecure", "unreliable", "non-compliant"]
    
    context = random.choice(contexts)
    return context.format(
        number=random.choice(numbers),
        aspect=random.choice(aspects),
        standard=random.choice(standards),
        event=random.choice(events),
        issue=random.choice(issues)
    )

def generate_output_format():
    return """1. Initial Setup:
• {setup1}
• {setup2}
• {setup3}

2. Configuration:
• {config1}
• {config2}
• {config3}

3. Verification:
• {verify1}
• {verify2}
• {verify3}"""

def generate_meraki_output(device, feature):
    outputs = {
        "MR46E": {
            "VLANs": {
                "setup1": "Configure SSID mappings",
                "setup2": "Set VLAN tags (10-Student, 20-Staff)",
                "setup3": "Enable VLAN pooling",
                "config1": "Configure RADIUS servers",
                "config2": "Set up access policies",
                "config3": "Enable L3 firewall rules",
                "verify1": "Test VLAN isolation",
                "verify2": "Verify client assignments",
                "verify3": "Monitor traffic segregation"
            },
            "QoS": {
                "setup1": "Enable QoS features",
                "setup2": "Configure DSCP tags",
                "setup3": "Set up traffic shaping",
                "config1": "Prioritize voice/video",
                "config2": "Set bandwidth limits",
                "config3": "Configure application QoS",
                "verify1": "Test VoIP quality",
                "verify2": "Monitor bandwidth usage",
                "verify3": "Validate QoS markers"
            }
        }
    }
    
    default_output = {
        "setup1": "Configure hardware settings",
        "setup2": "Set network parameters",
        "setup3": "Initialize security features",
        "config1": "Apply device policies",
        "config2": "Set up monitoring",
        "config3": "Enable logging",
        "verify1": "Test connectivity",
        "verify2": "Validate settings",
        "verify3": "Document changes"
    }
    
    return outputs.get(device, {}).get(feature, default_output)

def generate_google_output(service, feature):
    outputs = {
        "Google Classroom": {
            "sharing settings": {
                "setup1": "Configure domain sharing",
                "setup2": "Set class visibility",
                "setup3": "Define guardian access",
                "config1": "Set teacher permissions",
                "config2": "Configure student rights",
                "config3": "Enable external sharing",
                "verify1": "Test access levels",
                "verify2": "Validate permissions",
                "verify3": "Document policies"
            }
        }
    }
    
    default_output = {
        "setup1": "Configure service settings",
        "setup2": "Set user permissions",
        "setup3": "Enable required features",
        "config1": "Apply policies",
        "config2": "Set up monitoring",
        "config3": "Configure notifications",
        "verify1": "Test functionality",
        "verify2": "Validate access",
        "verify3": "Document setup"
    }
    
    return outputs.get(service, {}).get(feature, default_output)

def generate_training_data(num_entries):
    training_data = []
    
    # Generate Meraki examples
    for _ in range(num_entries // 4):
        device = random.choice(meraki_templates["devices"])
        feature = random.choice(meraki_templates["features"])
        template = random.choice(meraki_templates["instructions"])
        
        instruction = generate_instruction(template,
            device=device,
            location=random.choice(meraki_templates["locations"]),
            issue=random.choice(meraki_templates["issues"]),
            feature=feature,
            purpose=random.choice(meraki_templates["purposes"]),
            security=random.choice(meraki_templates["security"])
        )
        
        output_params = generate_meraki_output(device, feature)
        
        entry = {
            "instruction": instruction,
            "input": generate_input_context(),
            "output": generate_output_format().format(**output_params)
        }
        training_data.append(entry)
    
    # Generate Google Workspace examples
    for _ in range(num_entries // 4):
        service = random.choice(google_workspace_templates["services"])
        feature = random.choice(google_workspace_templates["features"])
        template = random.choice(google_workspace_templates["instructions"])
        
        instruction = generate_instruction(template,
            service=service,
            feature=feature,
            aspect=random.choice(google_workspace_templates["features"]),
            usergroup=random.choice(google_workspace_templates["usergroups"]),
            security=random.choice(google_workspace_templates["security"]),
            task=random.choice(google_workspace_templates["tasks"]),
            purpose=random.choice(google_workspace_templates["purposes"])
        )
        
        output_params = generate_google_output(service, feature)
        
        entry = {
            "instruction": instruction,
            "input": generate_input_context(),
            "output": generate_output_format().format(**output_params)
        }
        training_data.append(entry)
    
    # Generate macOS MDM examples
    for _ in range(num_entries // 4):
        template = random.choice(macos_mdm_templates["instructions"])
        instruction = generate_instruction(template,
            software=random.choice(macos_mdm_templates["software"]),
            device=random.choice(macos_mdm_templates["devices"]),
            setting=random.choice(macos_mdm_templates["settings"]),
            feature=random.choice(macos_mdm_templates["features"]),
            security=random.choice(macos_mdm_templates["security"]),
            usergroup=random.choice(google_workspace_templates["usergroups"]),
            task="deployment"
        )
        
        entry = {
            "instruction": instruction,
            "input": generate_input_context(),
            "output": generate_output_format().format(
                setup1="Configure MDM enrollment",
                setup2="Set security policies",
                setup3="Prepare software packages",
                config1="Deploy configurations",
                config2="Set up restrictions",
                config3="Configure updates",
                verify1="Test deployment",
                verify2="Verify compliance",
                verify3="Document settings"
            )
        }
        training_data.append(entry)
    
    # Generate Chrome OS examples
    for _ in range(num_entries // 4):
        template = random.choice(chrome_templates["instructions"])
        instruction = generate_instruction(template,
            feature=random.choice(chrome_templates["features"]),
            setting=random.choice(chrome_templates["settings"]),
            aspect=random.choice(chrome_templates["aspects"]),
            security=random.choice(chrome_templates["security"]),
            usergroup=random.choice(google_workspace_templates["usergroups"]),
            purpose=random.choice(google_workspace_templates["purposes"]),
            location=random.choice(meraki_templates["locations"]),
            software="Chrome apps"
        )
        
        entry = {
            "instruction": instruction,
            "input": generate_input_context(),
            "output": generate_output_format().format(
                setup1="Configure Chrome policies",
                setup2="Set enrollment settings",
                setup3="Prepare OU structure",
                config1="Deploy user settings",
                config2="Set up restrictions",
                config3="Configure updates",
                verify1="Test policies",
                verify2="Verify enrollment",
                verify3="Document configuration"
            )
        }
        training_data.append(entry)
    
    return training_data

def main():
    training_data = generate_training_data(1000)
    
    with open('training_data_expanded.json', 'w') as f:
        json.dump(training_data, f, indent=4)

if __name__ == "__main__":
    main()
