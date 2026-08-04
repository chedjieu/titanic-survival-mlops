terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.110"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "prefix" {
  type    = string
  default = "titanic-mlops"
}

variable "location" {
  type    = string
  default = "eastus"
}

resource "azurerm_resource_group" "rg" {
  name     = "${var.prefix}-rg"
  location = var.location
}

resource "azurerm_storage_account" "artifacts" {
  name                     = replace("${var.prefix}sa", "-", "")
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_eventhub_namespace" "eh" {
  name                = "${var.prefix}-ehns"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "Standard"
  capacity            = 1
}

resource "azurerm_eventhub" "passenger_events" {
  name                = "passenger.events"
  namespace_name      = azurerm_eventhub_namespace.eh.name
  resource_group_name = azurerm_resource_group.rg.name
  partition_count     = 3
  message_retention   = 1
}

resource "azurerm_eventhub" "survival_predictions" {
  name                = "survival.predictions"
  namespace_name      = azurerm_eventhub_namespace.eh.name
  resource_group_name = azurerm_resource_group.rg.name
  partition_count     = 3
  message_retention   = 1
}

output "resource_group" {
  value = azurerm_resource_group.rg.name
}

output "eventhub_namespace" {
  value = azurerm_eventhub_namespace.eh.name
}

output "artifact_storage_account" {
  value = azurerm_storage_account.artifacts.name
}
