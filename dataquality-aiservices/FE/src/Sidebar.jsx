import React from "react";

function Sidebar({
  items,
  selectedItem,
  onSelectItem,
  onSelectSubItem,
  renderLabel,
  layoutSizes,   // ⟵ WICHTIG!
}) {
  const { navbar, footer } = layoutSizes || { navbar: 0, footer: 0 };

const EXTRA_BOTTOM = 120;

const sidebarTop = navbar + 40;
const sidebarHeight = `calc(100vh - ${navbar + footer + EXTRA_BOTTOM}px)`;

  return (
    <div
      className="sidebar"
      style={{
        top: sidebarTop + "px",
        height: sidebarHeight,
      }}
    >
      {items.map((item) => (
        <div
          key={item.key}
          className={`sidebar-item ${selectedItem === item.key ? "selected" : ""}`}
        >
          <div
            onClick={() => onSelectItem(item.key)}
            className="sidebar-main"
          >
            {renderLabel(item.key)}
          </div>

          {selectedItem === item.key && item.subItems?.length > 0 && (
            <div className="sidebar-subitems">
              {item.subItems.map((subKey) => (
                <div
                  key={subKey}
                  className="sidebar-subitem"
                  onClick={() => onSelectSubItem(item.key, subKey)}
                >
                  {renderLabel(subKey, true)}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default Sidebar;
